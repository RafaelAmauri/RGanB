import torchvision
from torch.utils.data import Dataset
from utils import transform


class ColorizationDataset(Dataset):
    def __init__(self, imgPaths):

        self.imgPaths = imgPaths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        imgPath = self.imgPaths[idx]

        img   = torchvision.io.read_image(imgPath)
        l, ab = transform(img)
       
        return l, ab