# Style Transfer with Variable Alpha

Transform your personal photos into customized art posters using Neural Style Transfer (NST). This project blends content and style images with tunable artistic intensity, offering a cost-effective alternative to generic wall art.

## Project Overview

This system applies NST using a pre-trained VGG-19 model to combine the **content of one image** with the **artistic style of another**. Users can upload content and style images, control the style intensity via an alpha parameter (`style_weight`), and download the final stylized result.

Key Features:
- Upload content and style images (from URL or local path)
- Control style intensity (1e5 – 5e7)
- Choose optimizer: Adam or LBFGS
- Track loss and visualize intermediate results
- Save final stylized image in JPG format

## System Architecture

The project is modularized into three main classes:
- **ImageLoader**: Preprocesses input images
- **VGGFeatures**: Extracts content and style features from VGG-19 layers
- **StyleTransfer**: Implements NST training logic and visualizations

## Core Algorithm

- **Model**: Pre-trained VGG-19 on ImageNet
- **Framework**: PyTorch
- **Content Layer**: `conv4_2`
- **Style Layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- **Loss Functions**:
  - Content Loss (MSE)
  - Style Loss (Gram matrix + MSE)
- **Optimizers**:
  - Adam (faster, slightly noisier)
  - LBFGS (slower, smoother)

## Sample Results

<img width="332" alt="image" src="https://github.com/user-attachments/assets/148b3e65-863f-4402-9bf3-7c086e806d3b" />![image](https://github.com/user-attachments/assets/2bbe759e-1dd0-4f73-a52b-bae2443372a3)![image](https://github.com/user-attachments/assets/efc037f8-e852-4de1-9e0b-2f7dde43ecb1)![image](https://github.com/user-attachments/assets/ebe33ee7-790d-4c57-8783-a8a365b433dd)<img width="155" alt="image" src="https://github.com/user-attachments/assets/a0855d93-178e-4a4d-8b61-60b9e4640d28" />

Also, our model successfully swaps content and style roles to create flexible transformations
<img width="303" alt="image" src="https://github.com/user-attachments/assets/7646c2e3-f1a3-442b-ae7d-822802158b27" /><img width="303" alt="image" src="https://github.com/user-attachments/assets/1eb35094-a5e8-4ea0-a58c-97311d813065" />

We compared performance across optimizers and different alpha values:

| Optimizer | Style Weight | Total Loss | Content Loss | Style Loss |
|----------|---------------|------------|--------------|------------|
| Adam     | 1e5           | 0.9546     | 0.4408       | 5.14e-6    |
| Adam     | 5e7           | 25.7103    | 4.4712       | 4.2e-7     |
| LBFGS    | 1e5           | 0.9546     | 0.4408       | 5.14e-6    |
| LBFGS    | 5e7           | 25.7103    | 4.4712       | 4.2e-7     |

## Getting Started

```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
pip install -r requirements.txt
