 
# Deep Generative Models Unveil Patterns in Medical Images Through Vision-Language Conditioning

This is the official implementation of [the workshop paper](https://arxiv.org/abs/2410.13823) **(accepted by AIM-FM Workshop of NeurIPS2024🔥🔥🔥)**.

## 🌟 Abstract

Deep generative models have significantly advanced medical imaging analysis by enhancing dataset size and quality. Beyond mere data augmentation, our research in this paper highlights an additional, significant capacity of deep generative models: their ability to reveal and demonstrate patterns in medical images. We employ a generative structure with hybrid conditions, combining clinical data and segmentation masks to guide the image synthesis process. Furthermore, we innovatively transformed the tabular clinical data into textual descriptions. This approach simplifies the handling of missing values and also enables us to leverage large pre-trained vision-language models that investigate the relations between independent clinical entries and comprehend general terms, such as gender and smoking status. Our approach differs from and presents a more challenging task than traditional medical report-guided synthesis due to the less visual correlation of our clinical information with the images. To overcome this, we introduce a text-visual embedding mechanism that strengthens the conditions, ensuring the network effectively utilizes the provided information. Our pipeline is generalizable to both GAN-based and diffusion models. Experiments on chest CT, particularly focusing on the smoking status, demonstrated a consistent intensity shift in the lungs which is in agreement with clinical observations, indicating the effectiveness of our method in capturing and visualizing the impact of specific attributes on medical image patterns. Our methods offer a new avenue for the early detection and precise visualization of complex clinical conditions with deep generative models. 

## 🚀 Model Pipelines and Fusion Graphical Illustration:
![Overview of Method Pipeline1](./scr_README/methodver6.png)
![Overview of Method Pipeline2](./scr_README/fusionmethodsver6.png)
![Overview of Method Comparsion](./scr_README/comparsion.png)


## 🖼️ Illustrative Cases Demonstrating the Impact of Altered Prompt Content on Prediction Outcomes.

<table>
  <tr>
    <td align="center">
      <strong>(Default) Age: 68 Smoker: No 😊</strong><br>
      <img src="scr_README/readme_scr_gif/95__age_68_AIIB23_95_2th_crop_output.gif" alt="(Default) Age: 86 Smoker: Yes" width="100%">
    </td>
    <td align="center">
      <strong>Age: 24 Smoker: No 🧒</strong><br>
      <img src="scr_README/readme_scr_gif/95__age_24_AIIB23_95_2th_crop_output.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Difference in Voxel Values: Age 📈</strong><br>
      <img src="scr_README/readme_scr_gif/95_2th_crop_age_diff.gif" alt="Age Difference" width="80%">
    </td>
    <td align="center">
      <strong>Age: 68 Smoker: Yes 🚬</strong><br>
      <img src="scr_README/readme_scr_gif/95__smoker_yes_AIIB23_95_2th_crop_output.gif" alt="Smoker" width="100%">
    </td>
    <td align="center">
      <strong>Difference in Voxel Values: Smoker No vs Yes 🔄</strong><br>
      <img src="scr_README/readme_scr_gif/95_2th_crop_smoker_diff.gif" alt="Smoker Difference" width="80%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <strong>(Default) Age: 86 Smoker: Yes 🧓🚬</strong><br>
      <img src="scr_README/readme_scr_gif/150__age_86_AIIB23_150_0th_crop_output.gif" alt="(Default) Age: 86 Smoker: Yes" width="100%">
    </td>
    <td align="center">
      <strong>Age: 24 Smoker: Yes 🧒🚬</strong><br>
      <img src="scr_README/readme_scr_gif/150__age_24_AIIB23_150_0th_crop_output.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Difference in Voxel Values: Age 📊</strong><br>
      <img src="scr_README/readme_scr_gif/150_0th_crop_age_diff.gif" alt="Age Difference" width="80%">
    </td>
    <td align="center">
      <strong>Age: 86 Smoker: No 🚭</strong><br>
      <img src="scr_README/readme_scr_gif/150__smoker_no_AIIB23_150_0th_crop_output.gif" alt="Non-Smoker" width="100%">
    </td>
    <td align="center">
      <strong>Difference in Voxel Values: Smoker No vs Yes 🔁</strong><br>
      <img src="scr_README/readme_scr_gif/150_0th_crop_smoker_diff.gif" alt="Smoker Difference" width="80%">
    </td>
  </tr>
</table>


## 🎉 More Examples

<table>
  <tr>
    <td align="center">
      <strong>Ex1: (Default) Patient Status: Alive 😊</strong><br>
      <img src="scr_README/more examples/mortality/95_Alive+Dead_Alive_6_image.gif" alt="(Default) Patient Status: Alive" width="100%">
    </td>
    <td align="center">
      <strong>Patient Status: Dead 💀</strong><br>
      <img src="scr_README/more examples/mortality/95_Dead+Dead_Alive_6_image.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Difference in Voxel Values: Dead vs Alive 📉</strong><br>
      <img src="scr_README/more examples/mortality/95_Alive_vs_Dead_6_diff.gif" alt="Age Difference" width="80%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <strong>Ex2: (Default) Patient Status: Dead 💀</strong><br>
      <img src="scr_README/more examples/mortality/150_Dead+Dead_Alive_5_image.gif" alt="Patient Status: Dead" width="100%">
    </td>
    <td align="center">
      <strong>Patient Status: Alive 😊</strong><br>
      <img src="scr_README/more examples/mortality/150_Alive+Dead_Alive_5_image.gif" alt="Patient Status: Alive" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: Dead vs Alive 📊</strong><br>
      <img src="scr_README/more examples/mortality/150_Alive_vs_Dead_5_diff.gif" alt="Age Difference" width="80%">
    </td> 
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <strong>Ex3: (Default) DIAGNOSIS CODE: CTD-ILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/96_CTD-ILD_image.gif" alt="DIAGNOSIS CODE: CTD-ILD" width="100%">
    </td>
    <td align="center">
      <strong>DIAGNOSIS CODE: IPF 🩺</strong><br>
      <img src="scr_README/more examples/conditions/96_IPF_image.gif" alt="DIAGNOSIS CODE: IPF" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs IPF 📈</strong><br>
      <img src="scr_README/more examples/conditions/96__CTD-ILD_vs_IPF_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
        <td align="center">
      <strong>DIAGNOSIS CODE: UILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/96_UILD_image.gif" alt="DIAGNOSIS CODE: UILD" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs UILD 📉</strong><br>
      <img src="scr_README/more examples/conditions/96__CTD-ILD_vs_UILD_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: IPF vs UILD 📊</strong><br>
      <img src="scr_README/more examples/conditions/96_IPF_vs_UILD_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <strong>Ex4: (Default) DIAGNOSIS CODE: CTD-ILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/150_CTD-ILD_UILD_image.gif" alt="DIAGNOSIS CODE: CTD-ILD" width="100%">
    </td>
    <td align="center">
      <strong>DIAGNOSIS CODE: IPF 🩺</strong><br>
      <img src="scr_README/more examples/conditions/150_IPF_image.gif" alt="DIAGNOSIS CODE: IPF" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs IPF 📈</strong><br>
      <img src="scr_README/more examples/conditions/150_CTD-ILD_vs_IPF_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
        <td align="center">
      <strong>DIAGNOSIS CODE: UILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/150_UILD_image.gif" alt="DIAGNOSIS CODE: UILD" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs UILD 📉</strong><br>
      <img src="scr_README/more examples/conditions/150_CTD-ILD_vs_UILD_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: IPF vs UILD 📊</strong><br>
      <img src="scr_README/more examples/conditions/150_IPF_vs_UILD_diff.GIF" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <strong>Ex5: (Default) DIAGNOSIS CODE: CTD-ILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/1_96_CTD-ILD_image.gif" alt="DIAGNOSIS CODE: CTD-ILD" width="100%">
    </td>
    <td align="center">
      <strong>DIAGNOSIS CODE: IPF 🩺</strong><br>
      <img src="scr_README/more examples/conditions/1_96_IPF_image.gif" alt="DIAGNOSIS CODE: IPF" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs IPF 📈</strong><br>
      <img src="scr_README/more examples/conditions/1_96_CTD-ILD_vs_IPF_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
        <td align="center">
      <strong>DIAGNOSIS CODE: UILD 🏥</strong><br>
      <img src="scr_README/more examples/conditions/1_96_UILD_image.gif" alt="DIAGNOSIS CODE: UILD" width="100%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: CTD-ILD vs UILD 📉</strong><br>
      <img src="scr_README/more examples/conditions/1_96_CTD-ILD_vs_UILD_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
    <td align="center">
      <strong>Difference in Voxel Values: IPF vs UILD 📊</strong><br>
      <img src="scr_README/more examples/conditions/1_96_IPF_vs_UILD_diff.gif" alt="Difference in Voxel Values: DIAGNOSIS CODE" width="80%">
    </td> 
  </tr>
</table>


## 💡 Highlights

- **Conversion of Tabular Data into Text** 😊: This method efficiently addresses missing data issues and capitalizes on the capabilities of pre-trained vision-language models to decode clinical information.
- **Advanced Text Fusion Techniques** 🧠: We introduced techniques, including a cross-attention module and an Affine transformation fusion unit, to refine the conditioning process in cases where clinical information does not directly correspond to visual cues in images.
- **General Implementation for GAN and Diffusion Models** 🔄: Our pipeline is adaptable to both GAN-based and diffusion-based generative models.
## 🗂️ Code Structure

The code structure is organized as follows:

```bash
├── metrics_computation # Scripts for calculating evaluation metrics
├── models # Model definitions
├── options # Configuration and command line options
├── scr_README # README and documentation-related assets
├── utils # Utility functions and scripts 
├── generate_prompts.ipynb # Notebook for generating prompts
├── observe_difference.ipynb # Notebook for observing voxel differences 
├── inference_patch.py # Inference script (patch-level)
├── inference_whole.py # Inference script (whole-level)
└── train.py # Training script
```

## 📋 Requirements

Ensure you have the following dependencies installed:

```
apex==0.9.10dev
dominate==2.9.1
matplotlib==3.8.2
MedCLIP==0.0.3
monai==1.3.0
nibabel==5.2.1
numpy==1.26.4
pandas==2.2.1
Pillow==10.0.1
Pillow==10.2.0
pytorch_msssim==1.0.0
scikit_learn==1.4.0
scipy==1.12.0
SimpleITK==2.3.1
SimpleITK==2.3.1
tensorflow==2.15.0.post1
torch==2.1.2
torchmetrics==1.3.1
torchvision==0.16.2
tqdm==4.65.0
```

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```


## Citation

This repository is based on:

pix2pixHD: High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs ([code](https://github.com/NVIDIA/pix2pixHD) and 
[paper](https://arxiv.org/abs/1711.11585));

 

## 🔗 Citation

If you find our work interesting and useful, please consider citing:

```bibtex
@article{xing2024deep,
  title={Deep Generative Models Unveil Patterns in Medical Images Through Vision-Language Conditioning},
  author={Xing, Xiaodan and Ning, Junzhi and Nan, Yang and Yang, Guang},
  journal={arXiv preprint arXiv:2410.13823},
  year={2024}
}
```

## 📢 License

This project is licensed under the [MIT License](LICENSE).

 