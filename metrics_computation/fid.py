import torch
from torchmetrics.image.fid import FrechetInceptionDistance



def calculate_batch_3d_fid_score(real_images_batch, fake_images_batch, feature_dim=2048, seed=None):
    """
    Calculate the Frechet Inception Distance (FID) score for batches of 3D medical images, converting them
    to 3-channel images if necessary.

    Parameters:
    - real_images_batch (torch.Tensor): A batch of 3D real images with shape (B, C, D, H, W).
    - fake_images_batch (torch.Tensor): A batch of 3D generated (fake) images with shape (B, C, D, H, W).
    - feature_dim (int): The feature dimension for the Inception model to use. Default is 2048.
    - seed (int or None): Optional seed for reproducibility. Default is None.

    Returns:
    - float: The FID score.
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Initialize FID with the specified feature dimension
    fid = FrechetInceptionDistance(feature=feature_dim)

    # Reshape the batches to put the depth dimension as part of the batch
    # This converts the tensor from shape (B, C, D, H, W) to (B * D, C, H, W)
    B, C, D, H, W = real_images_batch.shape
    
    # Check if the images are single-channel, and if so, repeat the channel to make them 3-channel
    if C == 1:
        real_images_2d = real_images_batch.expand(-1, 3, -1, -1, -1).reshape(B * D, 3, H, W)
        fake_images_2d = fake_images_batch.expand(-1, 3, -1, -1, -1).reshape(B * D, 3, H, W)
    else:
        real_images_2d = real_images_batch.reshape(B * D, C, H, W)
        fake_images_2d = fake_images_batch.reshape(B * D, C, H, W)

    # Update FID with real and fake images
    fid.update(real_images_2d, real=True)
    fid.update(fake_images_2d, real=False)

    # Compute FID score
    fid_score = fid.compute()

    return fid_score.item()

    
if __name__ == "__main__":
    # Define the size of the 3D images and batch
    batch_size = 10
    channels = 1
    depth = 750  # Using smaller depth for example purposes
    height = 512  # Using smaller height for example purposes
    width = 512   # Using smaller width for example purposes

    # Generate a batch of random real and fake 3D medical images
    real_images_batch = torch.randint(0, 256, (batch_size, channels, depth, height, width), dtype=torch.uint8)
    fake_images_batch = torch.randint(0, 256, (batch_size, channels, depth, height, width), dtype=torch.uint8)

    # Calculate the average FID score for the batch of 3D images
    average_fid_score = calculate_batch_3d_fid_score(real_images_batch, fake_images_batch, feature_dim=192, seed=123)
    print("Average FID Score for Batch of 3D Images:", average_fid_score)