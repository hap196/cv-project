# Scene Segmentation with ADE20K Dataset

This project implements a semantic segmentation model for scene parsing using the ADE20K dataset. The model is built on a Fully Convolutional Network (FCN) architecture with different backbone options.

## Project Structure

```
.
├── ADE/                   # Dataset directory
├── models/                # Model implementations
│   ├── __init__.py
│   └── fcn.py             # FCN model implementation  
├── dataset_preparation.py # Dataset loading and preprocessing
├── train.py               # Training script
├── visualize.py           # Visualization of predictions
└── README.md              # Project documentation
```

## Setup and Requirements

### Environment Setup

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

The ADE20K dataset should be organized in the `ADE/` directory with the following structure:

```
ADE/
├── images/
│   └── ADE/
│       ├── training/
│       └── validation/
├── index_ade20k.pkl
└── objects.txt
```

## Usage

### Training

To train the model:

```bash
python train.py --backbone resnet50 --batch-size 8 --epochs 50 --lr 0.001
```

Optional arguments:
- `--backbone`: Backbone architecture (resnet18, resnet34, resnet50, vgg16)
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--device`: Device to use (cuda or cpu)
- `--weight-decay`: Weight decay for optimizer
- `--save-dir`: Directory to save checkpoints

### Visualization

To visualize model predictions:

```bash
python visualize.py --checkpoint checkpoints/model_best.pth --num-images 5
```

Optional arguments:
- `--checkpoint`: Path to the model checkpoint
- `--image-path`: Path to a single image (if not specified, uses validation images)
- `--num-images`: Number of validation images to visualize
- `--save-dir`: Directory to save visualizations
- `--device`: Device to use for inference

## Model Architecture

The project implements FCN (Fully Convolutional Network) with skip connections:

- **FCN32s**: Direct upsampling from the final layer
- **FCN16s**: Incorporates features from pool4
- **FCN8s**: Incorporates features from pool3 for more detailed segmentation

Available backbone options:
- ResNet18
- ResNet34
- ResNet50
- VGG16

## Evaluation Metrics

The model is evaluated using the following metrics:
- **Mean IoU (Intersection over Union)**: Average IoU across all classes
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Class Accuracy**: Accuracy for each class

## Acknowledgements

This project uses the ADE20K dataset as described in:

> Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene parsing through ADE20K dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 633-641).

## Custom Extensions

This project can be extended with the following features:
- Edge-aware segmentation
- Lightweight model design for mobile applications
- Instance-level semantic segmentation 