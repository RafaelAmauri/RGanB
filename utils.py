import torch
import numpy as np


def transform(img):
    img = (img / 255).to(torch.float32) # Normalize img in the [0, 1] range

    return img