import torchvision
from torch.utils.data import Dataset
from utils import transform
import torch

class ColorizationDataset(Dataset):
    def __init__(self, imgPaths):

        self.imgPaths = imgPaths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        imgBWPath, imgColorPath = self.imgPaths[idx]
        resize = torchvision.transforms.Resize(size=(256, 256))

        imgBW    = torchvision.io.read_image(imgBWPath, mode=torchvision.io.ImageReadMode.GRAY)
        imgBW    = resize(imgBW)
        imgBW    = transform(imgBW)

        imgColor = torchvision.io.read_image(imgColorPath)
        imgColor = resize(imgColor)
        imgColor = transform(imgColor)


        return imgBW, imgColor