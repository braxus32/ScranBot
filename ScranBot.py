# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
