import os
import math
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
    """
    Save a 3D volume tensor as a series of 2D images along the last dimension.
    Args:
    - volume_tensor: 3D tensor representing the volume, with shape [Depth, Height, Width].
    - image_id: Identifier for the volume.
    - output_dir: Directory where the images will be saved.
    """
    # Remove any singleton dimensions (especially if tensor has shape [1, 1, Depth, Height, Width])
    volume_tensor = volume_tensor.squeeze()
    # print('volume_tensor: ', volume_tensor.shape)

    # Verify that we now have a 3D tensor after squeezing
    if len(volume_tensor.shape) != 3:
        raise ValueError("The input tensor should be 3D after removing singleton dimensions.")

    # Create a folder for the current image_id
    image_folder = os.path.join(output_dir, f"{image_id}_png_images_{suffix}")
    os.makedirs(image_folder, exist_ok=True)

    # Iterate through the last dimension (depth) of the volume tensor
    for slice_index in range(volume_tensor.shape[-1]):
        print(f'Generating slice {slice_index}th image of {suffix}...', end='\r')
        # Get a single slice from the volume along the last dimension
        slice_tensor = volume_tensor[:, :, slice_index]
        # Save each slice as an image
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


def prepare_batch(opt, image, ijk_patch_indices):
    image_batches = []
    for patch in ijk_patch_indices:
        image_patch = image[..., patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
        image_batches.append(image_patch)

    return image_batches




def inference(opt, model, dataset, prompt, dataloader = None):


    if opt.single_file is True: 
        model.eval()
        with torch.no_grad():
            data = dataset.get_one_instance(
            opt.image_path, opt.label_path, opt.file_name) 
            whole_label = data["label"]
            image_id = data['image_id']
            # 1 ---------- creating the batches from the data  --------------------

            # a weighting matrix will be used for averaging the overlapped region
            max_x, max_y, max_z = data["image"][0,:,:].shape[0], data["image"][0,:,:,:].shape[1], data["image"][0,:,:,:].shape[2]

            weight_np = torch.zeros((max_x, max_y, max_z))
            predictions = torch.zeros((max_x, max_y, max_z))
            # prepare image batch indices
            inum = int(
                math.ceil((weight_np.shape[0] - opt.patch_size[0]) / float(opt.stride_inplane))) + 1
            jnum = int(
                math.ceil((weight_np.shape[1] - opt.patch_size[1]) / float(opt.stride_inplane))) + 1
            knum = int(
                math.ceil((weight_np.shape[2] - opt.patch_size[2]) / float(opt.stride_layer))) + 1


            patch_total = 0
            ijk_patch_indices = []

            

            for i in range(inum):
                for j in range(jnum):
                    for k in range(knum):
                        # print(f'i: {i}, j: {j}, k: {k}')

                        # if patch_total % opt.batch_size == 0:
                        #     ijk_patch_indicies_tmp = []

                        istart = i * opt.stride_inplane
                        if istart + opt.patch_size[0] > max_x:  # for last patch
                            istart = max_x - opt.patch_size[0]
                        iend = istart + opt.patch_size[0]

                        jstart = j * opt.stride_inplane
                        if jstart + opt.patch_size[1] > max_y:  # for last patch
                            jstart = max_y - opt.patch_size[1]
                        jend = jstart + opt.patch_size[1]

                        kstart = k * opt.stride_layer
                        if kstart + opt.patch_size[2] > max_z:  # for last patch
                            kstart = max_z - opt.patch_size[2]
                        kend = kstart + opt.patch_size[2]

                        ijk_patch_indices.append(
                            [istart, iend, jstart, jend, kstart, kend])

                        # if patch_total % opt.batch_size == 0:
                        #     ijk_patch_indices.append(ijk_patch_indicies_tmp)

                        patch_total += 1
 
            if opt.model not in opt.label2maskmethods:  
                batches = prepare_batch(opt, data["label"][0,:,:,:], ijk_patch_indices)
            else:
                batches = prepare_batch(opt, data["labelmask"][:,:,:,:], ijk_patch_indices)

            # 1 ---------- creating the batches from the data  --------------------

            # 2 ---------- obtain the predictions from the batches   --------------------

            print('weight_np: ', weight_np.shape)

            for i in tqdm(range(len(batches))):
                batch = batches[i]
                # print('batch: ', len(batch))
 
                if opt.model not in opt.label2maskmethods:
                    data["label"] = batch.unsqueeze(0).unsqueeze(0)
                else:
                    data["labelmask"] = batch.unsqueeze(0)

                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                prediction = visuals['fake_B'].squeeze(0)

                istart = ijk_patch_indices[i][0]
                iend = ijk_patch_indices[i][1]
                jstart = ijk_patch_indices[i][2]
                jend = ijk_patch_indices[i][3]
                kstart = ijk_patch_indices[i][4]
                kend = ijk_patch_indices[i][5]
                predictions[istart:iend, jstart:jend, kstart:kend] += prediction[0,:,:,:].cpu()
                weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

            average_predictions = (predictions / weight_np) + 0.01
            # 2 ---------- obtain the predictions from the batches   --------------------
            

            # 3 ----------  save the results    --------------------

            if opt.save_3Dvolume is True:
                save_as_nifti(average_predictions, f"{opt.results_dir}/{image_id}.nii.gz")

            save_volume_as_images(
                average_predictions, opt.results_dir, f'{image_id}',suffix="prediction", mode="G")
            save_volume_as_images(
                data["image"], opt.results_dir, f'{image_id}',suffix="original",mode="G")
            save_volume_as_images(
                whole_label, opt.results_dir, f'{image_id}', suffix="label", mode="RGB")

            # 3 ----------  save the results    --------------------
    else:
        print("Folder Predictions Mode!")

        model.eval()
        with torch.no_grad():
            for index_file, data_indexer in enumerate(dataloader):
                # 1 ---------- obtain the data from the dataloader  -------------------- 

                data = dataset.get_one_instance(
                    data_indexer["data_path"][0], data_indexer["label_path"][0], data_indexer["file_name"][0])

                data["image"] = data["image"].unsqueeze(0)
                data["label"] = data["label"].unsqueeze(0) 
                whole_label = data["label"]
                image_id = data['image_id'] 

                if opt.prompt_customised is True and "prompt" in data:
                    data['prompt'] = [prompt] 

                # 1 ---------- obtain the data from the dataloader  --------------------


                # 2 ---------- creating the batches from the data  --------------------

                # a weighting matrix will be used for averaging the overlapped region
                max_x, max_y, max_z = data["image"][0,0,:,:,:].shape[0], data["image"][0,0,:,:,:].shape[1], data["image"][0,0,:,:,:].shape[2] 
                weight_np = torch.zeros((max_x, max_y, max_z))
                predictions = torch.zeros((max_x, max_y, max_z))
                inum = int(
                    math.ceil((weight_np.shape[0] - opt.patch_size[0]) / float(opt.stride_inplane))) + 1
                jnum = int(
                    math.ceil((weight_np.shape[1] - opt.patch_size[1]) / float(opt.stride_inplane))) + 1
                knum = int(
                    math.ceil((weight_np.shape[2] - opt.patch_size[2]) / float(opt.stride_layer))) + 1


                patch_total = 0
                ijk_patch_indices = []

                for i in range(inum):
                    for j in range(jnum):
                        for k in range(knum):
                            # print(f'i: {i}, j: {j}, k: {k}')

                            # if patch_total % opt.batch_size == 0:
                            #     ijk_patch_indicies_tmp = []

                            istart = i * opt.stride_inplane
                            if istart + opt.patch_size[0] > max_x:  # for last patch
                                istart = max_x - opt.patch_size[0]
                            iend = istart + opt.patch_size[0]

                            jstart = j * opt.stride_inplane
                            if jstart + opt.patch_size[1] > max_y:  # for last patch
                                jstart = max_y - opt.patch_size[1]
                            jend = jstart + opt.patch_size[1]

                            kstart = k * opt.stride_layer
                            if kstart + opt.patch_size[2] > max_z:  # for last patch
                                kstart = max_z - opt.patch_size[2]
                            kend = kstart + opt.patch_size[2]

                            ijk_patch_indices.append(
                                [istart, iend, jstart, jend, kstart, kend])

                            # if patch_total % opt.batch_size == 0:
                            #     ijk_patch_indices.append(ijk_patch_indicies_tmp)

                            patch_total += 1

                if opt.model not in opt.label2maskmethods:  
                    batches = prepare_batch(opt, data["label"][0,0,:,:,:], ijk_patch_indices)
                else:
                    batches = prepare_batch(opt, data["labelmask"][:,:,:,:], ijk_patch_indices)



                # 2 ---------- creating the batches from the data  --------------------




                # 3 ---------- obtain the predictions from the batches   --------------------  

                for i in tqdm(range(len(batches))):
                    batch = batches[i]
                    # print('batch: ', len(batch))
                    if opt.model not in opt.label2maskmethods:
                        data["label"] = batch.unsqueeze(0).unsqueeze(0)
                    else:
                        data["labelmask"] = batch.unsqueeze(0)


                    model.set_input(data)
                    model.test()
                    visuals = model.get_current_visuals()
                    prediction = visuals['fake_B'].squeeze(0)

                    istart = ijk_patch_indices[i][0]
                    iend = ijk_patch_indices[i][1]
                    jstart = ijk_patch_indices[i][2]
                    jend = ijk_patch_indices[i][3]
                    kstart = ijk_patch_indices[i][4]
                    kend = ijk_patch_indices[i][5]
                    predictions[istart:iend, jstart:jend, kstart:kend] += prediction[0,:,:,:].cpu()
                    weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

                average_predictions = (predictions / weight_np) + 0.01
                # 3 ---------- obtain the predictions from the batches   --------------------
                

                # 4 ----------  save the results    -------------------- 
                if opt.save_3Dvolume is True:
                    save_as_nifti(average_predictions, f"{opt.results_dir}/{image_id}.nii.gz")

                save_volume_as_images(
                    average_predictions, opt.results_dir, f'{image_id}',suffix="prediction", mode="G")
                save_volume_as_images(
                    data["image"], opt.results_dir, f'{image_id}',suffix="original",mode="G")
                save_volume_as_images(
                    whole_label, opt.results_dir, f'{image_id}', suffix="label", mode="RGB")

                # 4 ----------  save the results    --------------------
                    
            
    



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
    opt.workers = 4
    opt.label2maskmethods = ["medddpm", "medddpmtext", "medddpmtextcross", "medddpmvisualprompt"] # Specific Diffusion special processing
 
    # User Case Description:
    # When running the inference script, the parameters that must be specified include: --name, --netG, and --model. These parameters are essential for execution, and their values should match the corresponding entries in the checkpoint directory.

    # Command Example:
    # python inference_single.py --name AWVS_200*64_pix2pix_improved_promopts_text_embeddings_lr_0.0001_ver1 --model pix2pixclip --netG resnet_9blocks_with_text_encoderembeddings_lr_0.0001_ver1

    # The parameters must be compatible with the model training configuration. For details, please refer to the 'opt' file in the checkpoint directory.

    # In addition, the following are the parameters that need to be adjusted within the Python text file
    # Inference Configuration:
    opt.patch_size = (256, 256, 64)  # The patch size used for inference. Needs to match the size in the model checkpoint.
    opt.input_nc = 1
    opt.save_3Dvolume = False # To save the file in nii.gz format
    opt.single_file = False  # If False, read all files in the path
    # Custom Prompt Evaluation Options:
    opt.prompt_customised = False  # Set to False to use the original pre-extracted prompts. When set to True, allows for custom prompt evaluation.


    opt.stride_inplane = 128 # Overlapping reconstruction method
    opt.stride_layer = 32 # Overlapping reconstruction method

    prompt = None # only be used when opt.prompt_customised is set to be true

    # Select a single NIfTI (.nii.gz) file for imaging:
    # Ensure the selected file and directory exist before proceeding.
    opt.file_name = "AIIB23_169.nii.gz"

    # Single File mode
    visualisation_root = "../../data/3D-CycleGan-Pytorch_data/"
    opt.image_path = f"{visualisation_root}/img/{opt.file_name}"
    opt.label_path = f"{visualisation_root}/gtlung/{opt.file_name}"

    # Result Configuration:
    # Define a suffix to distinguish the output directory name, and set the directory to save the results.
    customised_name = "whole_136-140"
    opt.suffix = f'{customised_name}' + opt.which_epoch + f"_epochs_whole_stride_{opt.stride_inplane}_{opt.stride_layer}"

    if opt.single_file is True:
        opt.suffix = "_" + opt.file_name.split(".")[0] + "_" + opt.suffix
    opt.results_dir = os.path.join('./results/', opt.name  + opt.suffix)

 
    os.makedirs(opt.results_dir, exist_ok=True)
    if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
        opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        torch.cuda.set_device(opt.device)
    else:
        opt.device = torch.device("cpu")
 
    model = create_model(opt) 
    model.setup(opt)
    model.load_networks(opt.which_epoch)
  
 
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 50))
    Transforms = None  # Define any required transforms here
 

    # Create the dataset and dataloader for inference 
    dataset = NiftiPromptDataset(opt.data_path, path_A=opt.generate_path, path_B=opt.input_path,
                                 which_direction='AtoB', transforms=Transforms,
                                 train=True, test = True, label2mask=True if opt.model in opt.label2maskmethods else False)
    if opt.single_file is False:
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=opt.workers)
    else: 
        dataloader = None
   

    # Perform inference using the dataloader
    inference(opt, model, dataset, dataloader = dataloader,  prompt=prompt)
