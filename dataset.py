from torch.utils.data import Dataset
from utils import transform
from skimage import io

class ColorizationDataset(Dataset):
    def __init__(self, imgPaths):

        self.imgPaths = imgPaths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        imgBWPath, imgColorPath = self.imgPaths[idx]

        imgColor = io.imread(imgColorPath)
        imgColorL, imgColorAB, imgColorLAB  = transform(imgColor)
        
        return imgColorL, imgColorAB, imgColorLAB