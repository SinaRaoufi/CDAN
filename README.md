# CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement
[Hossein Shakibania](https://scholar.google.com/citations?user=huveR90AAAAJ&hl=en&authuser=1), [Sina Raoufi](https://scholar.google.com/citations?user=f0iw8XsAAAAJ&hl=en&authuser=1), and [Hassan Khotanlou](https://scholar.google.com/citations?user=5YX31NgAAAAJ&hl=en&authuser=1)

[![paper](https://img.shields.io/badge/arXiv-Paper-red)](https://doi.org/10.48550/arXiv.2308.12902)
[![paper](https://img.shields.io/badge/papers_with_code-Paper-blue)](https://paperswithcode.com/paper/cdan-convolutional-dense-attention-guided)
<p align="justify">
<strong>Abstract:</strong> Low-light images, characterized by inadequate illumination, pose challenges of diminished clarity, muted colors, and reduced details. Low-light image enhancement, an essential task in computer vision, aims to rectify these issues by improving brightness, contrast, and overall perceptual quality, thereby facilitating accurate analysis and interpretation. This paper introduces the Convolutional Dense Attention-guided Network (CDAN), a novel solution for enhancing low-light images. CDAN integrates an autoencoder-based architecture with convolutional and dense blocks, complemented by an attention mechanism and skip connections. This architecture ensures efficient information propagation and feature learning. Furthermore, a dedicated post-processing phase refines color balance and contrast. Our approach demonstrates notable progress compared to state-of-the-art results in low-light image enhancement, showcasing its robustness across a wide range of challenging scenarios. Our model performs remarkably on benchmark datasets, effectively mitigating under-exposure and proficiently restoring textures and colors in diverse low-light scenarios. This achievement underscores CDAN's potential for diverse computer vision tasks, notably enabling robust object detection and recognition in challenging low-light conditions.
</p>
  
<p class="row" float="left" align="middle">
<img style="width: 100%; height: auto;" src="assets/cdan_model.jpeg"/>
</p>
<p align="center"><b>Figure 1:</b> The overall structure of the proposed model.</p>

## Experimental Results
<p align="justify">
In this section, we present the experimental results obtained by training our CDAN model using the LOw-Light (LOL) dataset and evaluating its performance on multiple benchmark datasets. The purpose of this evaluation is to assess the robustness of our model across a spectrum of challenging lighting conditions. 
</p>

### Datasets
| Dataset        | No. of Images | Paired | Characteristics        |
|----------------|---------------|--------|-------------------------|
| [LOL](https://paperswithcode.com/dataset/lol)        | 500           | :white_check_mark: | Indoor |
| [ExDark](https://paperswithcode.com/dataset/exdark)     | 7363          | :x: | Extremely Dark, Indoor, Outdoor |
| [DICM](https://paperswithcode.com/dataset/dicm)       | 69           | :x: | Indoor, Outdoor |
| [VV](https://sites.google.com/site/vonikakis/datasets?authuser=0)         | 24          | :x: | Severely under/overexposed areas|

### Quantitative Evaluation
| Learning method      | Method                      | Avg. PSNR ↑  | Avg. SSIM ↑ | Avg. LPIPS ↓ |
|----------------------|-----------------------------|-------------|-------------|--------------|
| Supervised           | LLNET                       | 17.959      | 0.713       | 0.360        |
| Supervised           | LightenNet                  | 10.301      | 0.402       | 0.394        |
| Supervised           | MBLLEN                      | 17.902      | 0.715       | 0.247        |
| Supervised           | Retinex-Net                 | 16.774      | 0.462       | 0.474        |
| Supervised           | KinD                        | 17.648      | 0.779       | 0.175        |
| Supervised           | Kind++                      | 17.752      | 0.760       | 0.198        |
| Supervised           | TBEFN                       | 17.351      | 0.786       | 0.210        |
| Supervised           | DSLR                        | 15.050      | 0.597       | 0.337        |
| Supervised           | LAU-Net                     | 21.513      | 0.805       | 0.273        |
| Semi-supervised      | DRBN                        | 15.125      | 0.472       | 0.316        |
| Unsupervised         | EnlightenGAN                | 17.483      | 0.677       | 0.322        |
| Zero-shot            | ExCNet                      | 15.783      | 0.515       | 0.373        |
| Zero-shot            | Zero-DCE                    | 14.861      | 0.589       | 0.335        |
| Zero-shot            | RRDNet                      | 11.392      | 0.468       | 0.361        |
|                      | Proposed (CDAN)             | 20.102      | 0.816       | 0.167        |

### Qualitative Evaluation
<p class="row" float="left" align="middle">
<img style="width: 100%; height: auto;" src="assets/exdark.jpeg"/>
</p>
<p align="center"><b>Figure 2:</b> Visual comparison of state-of-the-art models on ExDark dataset.</p>

<p class="row" float="left" align="middle">
<img style="width: 100%; height: auto;" src="assets/dicm.jpeg"/>
</p>
<p align="center"><b>Figure 3:</b> Visual comparison of state-of-the-art models on DICM dataset.</p>


## Installation

To get started with the CDAN project, follow these steps:

### 1. Clone the Repository

You can clone the repository using Git. Open your terminal and run the following command:

```bash
git clone https://github.com/your-username/cdan-project.git
```
### 2. Configure Environmental Variables
<p align="justify">
After cloning, navigate to the project directory and locate the .env file. This file contains important hyperparameter values and configurations for the CDAN model. You can customize these variables according to your requirements.
</p>

Open the .env file using a text editor of your choice and modify the values as needed:
```
# Example .env file

# Directory paths
DATA_DIR=/path/to/your/data/directory
MODEL_DIR=/path/to/your/model/directory

# Hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=32
```
### 3. Install Dependencies

You can install project dependencies using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Project

You are now ready to run the CDAN project. To start the training, use the following command:

```bash
python train.py
```

To test the trained model, run:
```bash
python test.py
```

## Requirements
The following hardware and software were used for training the model:
- GPU: NVIDIA GeForce RTX 3090
- RAM: 24 GB SSD
- Operating System: Ubuntu 22.04.2 LTS
- Python version: 3.9.15
- PyTorch version: 2.0.1
- PyTorch CUDA version: 11.7


## Citation
```
@misc{shakibania2023cdan,
      title={CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement}, 
      author={Hossein Shakibania and Sina Raoufi and Hassan Khotanlou},
      year={2023},
      eprint={2308.12902},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
