import torch
from torch.utils.data import Dataset
from utils import transform, undoTransform
from skimage import io
import pandas as pd
import numpy as np
import skimage

class ColorizationDataset(Dataset):
    def __init__(self, imgPaths):

        self.imgPaths = imgPaths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        imgBWPath, imgColorPath = self.imgPaths[idx]

        imgColor = io.imread(imgColorPath)
        imgColorL, imgColorAB = transform(imgColor)
        
        return imgColorL, imgColorAB