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

### Qualitative Evaluation

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



## Citation
