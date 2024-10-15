import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F 
from inspect import isfunction
from functools import partial

from tqdm.auto  import tqdm



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        with_condition = True,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0,
        betaschedule = "cosine"
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            # betas = cosine_beta_schedule(timesteps)
            betas = get_named_beta_schedule(timesteps, schedule_name=betaschedule)
            

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c = None,**kwargs):
        if self.with_condition:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t,**kwargs))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False,**kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised,**kwargs)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors = None,**kwargs):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.with_condition:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, condition_tensors=condition_tensors,**kwargs)
            else:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 2, condition_tensors = None,**kwargs):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors = condition_tensors,**kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors = None, noise = None,**kwargs):
        b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.with_condition:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) 
            x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t, **kwargs)
      
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        x = x.transpose(4, 2)
        print("----------------------------->  x.shape",*x.shape)
        print("----------------------------->  condition_tensors.shape",*condition_tensors.shape)
        condition_tensors = condition_tensors.transpose(4, 2)
        print("----------------------------->  x.shape",*x.shape)
        print("----------------------------->  condition_tensors.shape",*condition_tensors.shape)
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size
        assert h == img_size and w == img_size and d == depth_size, f'Expected dimensions: height={img_size}, width={img_size}, depth={depth_size}. Actual: height={h}, width={w}, depth={d}.'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)





def get_named_beta_schedule( num_diffusion_timesteps, schedule_name = "cosine"):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L18
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return cosine_beta_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")



def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)