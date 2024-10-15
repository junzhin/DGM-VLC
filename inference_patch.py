import os
import random
import torch
import nibabel as nib
from models import create_model
from PIL import Image
import numpy as np
from options.train_options import TrainOptions
from utils.NiftiPromptDataset import NiftiPromptDataset 
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  



def seed_everything(seed):
    """
    Seed all necessary random number generators to ensure reproducible results.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA (for a single GPU)
        torch.cuda.manual_seed_all(seed)  # PyTorch CUDA (for all GPUs)


def save_volume_as_images(volume_tensor, output_dir, image_id, mode, suffix="" ):
 
    volume_tensor = volume_tensor.squeeze()

    if len(volume_tensor.shape) != 3:
        raise ValueError("The input tensor should be 3D after removing singleton dimensions.")

    # Create a folder for the current image_id
    image_folder = os.path.join(output_dir, f"{image_id}_png_images_{suffix}")
    os.makedirs(image_folder, exist_ok=True)

    for slice_index in range(volume_tensor.size(-1)):
        print('slice_index: ', slice_index)
        slice_tensor = volume_tensor[:, :, slice_index]
        save_slice_as_image(slice_tensor, image_folder,
                            image_id, slice_index, mode=mode)


def save_slice_as_image(slice_tensor, output_dir, image_id, slice_index, mode="G"):
    """
    Save a 2D tensor slice as an image.
    
    Args:
    - slice_tensor: 2D tensor representing the slice.
    - output_dir: Directory where the images will be saved.
    - image_id: Identifier for the volume.
    - slice_index: Index of the slice in the volume.
    - mode: Mode in which the image should be saved ("G" for grayscale, "RGB" for RGB mode).
    """

    slice_tensor = slice_tensor.transpose(0, 1)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if mode == "G":
        # Normalize the grayscale slice to the range [0, 255]
        normalized_slice = slice_tensor - slice_tensor.min()
        if normalized_slice.max() > 0:
            normalized_slice = normalized_slice * \
                (255.0 / normalized_slice.max())
        img_array = normalized_slice.cpu().detach().numpy().astype(np.uint8)
        # 'L' mode for grayscale image
        slice_image = Image.fromarray(img_array, mode='L')

    elif mode == "RGB":
        # Find the unique values in the tensor
        unique_values = torch.unique(slice_tensor)
        # print('unique_values: ', unique_values)
 

        color_map = {
            -1.0000: [0, 0, 0],   # Black color: background 
            -0.5000: [0, 255, 0],   # Green color: right,left lung
            0.0000: [255, 0, 0],  # Red color: right,left lung
            0.5000: [255, 255, 0]  # Yellow color airway
        }

        # Create an empty array for the RGB image
        rgb_array = np.zeros((*slice_tensor.shape, 3), dtype=np.uint8)

        # Fill in the array with the mapped colors
        for value, color in color_map.items():
            mask = slice_tensor == value
            rgb_array[mask.cpu().numpy()] = color

        slice_image = Image.fromarray(rgb_array, mode='RGB')

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Save the image
    slice_image.save(os.path.join(
        output_dir, f"{image_id}_{slice_index:04d}.png"))

def save_as_nifti(tensor, filename):
    np_image = tensor.cpu().squeeze().numpy()
    nifti_image = nib.Nifti1Image(np_image, affine=np.eye(4))
    nib.save(nifti_image, filename)


def inference(opt, model, dataset, prompt, dataloader = None):


    if opt.single_file is True:
        print("Singe File Predictions!")
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(opt.ncrops)):
                data = dataset.get_one_instance(
                opt.image_path, opt.label_path, opt.file_name)
                data["image"] = data["image"].unsqueeze(0)
                data["label"] = data["label"].unsqueeze(0) 
                image_id = data['image_id'] 

                if opt.prompt_debug is True and "prompt" in data:
                    data['prompt'] = [prompt]
                    print('[prompt]: ', [prompt])

                model.set_input(data) 

                visuals = model.get_current_visuals()
                prediction = visuals['fake_B'].squeeze(0)

                if opt.save_3Dvolume is True:
                    save_as_nifti(
                    prediction, f"{opt.results_dir}/{image_id}_{i}th_crop_output.nii.gz")

                save_volume_as_images(
                    prediction, opt.results_dir, f'{image_id}_{i}th_crop',suffix="prediction", mode="G")
                save_volume_as_images(
                    data["image"], opt.results_dir, f'{image_id}_{i}th_crop',suffix="original",mode="G")
                save_volume_as_images(
                    data["label"], opt.results_dir, f'{image_id}_{i}th_crop', suffix="label", mode="RGB")
    else:
        print("Folder Predictions!")
        model.eval()
        with torch.no_grad():
            for index_file, data_indexer in enumerate(dataloader):
                for i in tqdm(range(opt.ncrops)):
                     

                    data = dataset.get_one_instance(
                        data_indexer["data_path"][0], data_indexer["label_path"][0], data_indexer["file_name"][0])

                    data["image"] = data["image"].unsqueeze(0)
                    data["label"] = data["label"].unsqueeze(0) 
                    image_id = data['image_id'] 


                    if opt.prompt_debug is True and "prompt" in data:
                        data['prompt'] = [prompt]
                        print('[prompt]: ', [prompt])

                    model.set_input(data) 

                    visuals = model.get_current_visuals()
                    prediction = visuals['fake_B'].squeeze(0)

                    if opt.save_3Dvolume is True:
                        save_as_nifti(prediction, f"{opt.results_dir}/{image_id}_{i}th_crop_output.nii.gz")

                    save_volume_as_images(
                        prediction, opt.results_dir, f'{image_id}_{i}th_crop',suffix="prediction", mode="G")
                    save_volume_as_images(
                        data["image"], opt.results_dir, f'{image_id}_{i}th_crop',suffix="original",mode="G")
                    save_volume_as_images(
                        data["label"], opt.results_dir, f'{image_id}_{i}th_crop', suffix="label", mode="RGB")
            
    



seed = 50   
seed_everything(seed) 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



if __name__ == '__main__':
    # Parse options using TrainOptions
    opt = TrainOptions().parse()

    opt.epoch_count = 0
    opt.which_epoch = "latest"
    opt.data_path = "./test_small_file_ver2/"
   
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.workers = 8
    opt.label2maskmethods = ["medddpm", "medddpmtext", "medddpmtextcross", "medddpmvisualprompt"] # Specific Diffusion special processing

    # -----------------------------------------------Editable Section------------------------------------------------

    # User Case Description:
    # When running the inference script, it is mandatory to specify parameters including: --name, --netG, and --model. These parameters are necessary for execution, and their values must match with the corresponding entries in the checkpoint directory.

    # Command Example:
    # python inference_single.py --name AWVS_200*64_pix2pix_improved_promopts_text_embeddings_lr_0.0001_ver1 --model pix2pixclip --netG resnet_9blocks_with_text_encoderembeddings_lr_0.0001_ver1

    # Parameters need to be compatible with the model training configuration. For details, refer to the 'opt' file in the checkpoint directory.

    # Besides, the following are parameters that need to be adjusted inside the Python text file
    # Inference Configuration:
    opt.patch_size = (256, 256, 64)  # Patch size used for inference. Must match the size in the model checkpoint.
    opt.ncrops = 5 # Number of patches to generate for demonstration purposes.
    opt.save_3Dvolume = True # To save files in nii.gz format
    opt.single_file = False  # If False, read all files in the path
    # Custom Prompt Evaluation Options:
    opt.prompt_debug = False  # Set to False to use original pre-extracted prompts. When set to True, custom prompt evaluation is allowed.

    prompt = None 
    opt.file_name = "AIIB23_95.nii.gz"
 
    visualisation_root = "./test_small_file_ver2/" # Image generation for standalone file mode
    opt.image_path = f"{visualisation_root}/img/{opt.file_name}"
    opt.label_path = f"{visualisation_root}/gtlung/{opt.file_name}"

    # Result Configuration:
    # Define a suffix to differentiate the output directory name and set the directory to save the results.
    opt.customised_name = "_patch_91-171_"
    opt.suffix = f'{opt.customised_name}' + opt.which_epoch + "_epochs"

    if opt.single_file is True:
        opt.suffix = "_" + opt.file_name.split(".")[0] + "_" + opt.suffix 
    opt.results_dir = os.path.join('./results/', opt.name + opt.suffix)

    # -----------------------------------------------Editable Section------------------------------------------------
 
    os.makedirs(opt.results_dir, exist_ok=True)
    if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
        opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        torch.cuda.set_device(opt.device)
    else:
        opt.device = torch.device("cpu")
 
    model = create_model(opt) 
    model.setup(opt)
    model.load_networks(opt.which_epoch)
  
 
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 3))
    # min_pixel = 10
    Transforms = [
        NiftiDataset.Padding(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
        NiftiDataset.RandomCrop(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    ]

    # Create the dataset and dataloader for inference
    inference_transforms = []  # Define any required transforms here
    dataset = NiftiPromptDataset(opt.data_path, path_A=opt.generate_path, path_B=opt.input_path,
                                 which_direction='AtoB', transforms=Transforms,
                                 train=True, test=True, label2mask=True if opt.model in opt.label2maskmethods else False)
    if opt.single_file is False:
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=8)
    else: 
        dataloader = None
   

    # Perform inference using the dataloader
    inference(opt, model, dataset, dataloader = dataloader,  prompt=prompt)
