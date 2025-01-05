import torch
import numpy as np

from skimage.color import rgb2lab, lab2rgb


def transform(img):
    img = img.permute(1, 2, 0)
    img = torch.from_numpy(rgb2lab(img))
    img = img.permute(2, 0, 1)

    l  = img[0, :, :]  # The l component is the first channel
    ab = img[1:, :, :] # The ab component is the first and second channel


    l = l.unsqueeze(0) # Shape (1, H, W)

    l  = l.to(torch.float32)
    ab = ab.to(torch.float32)

    return l, ab


def reverseTransform(l, ab):
    # Combine L and AB into one LAB image
    lab_image = np.concatenate([l, ab], axis=0)  # Shape (2, H, W) -> (H, W, 3)

    # Convert the LAB image back to RGB
    rgb_image = lab2rgb(lab_image.transpose(1, 2, 0))  # Convert to shape (H, W, 3) for RGB

    # Rescale to the [0, 255] range and convert to uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image 