from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self, valuesL, valuesAB):

        self.valuesL  = valuesL
        self.valuesAB = valuesAB

    def __len__(self):
        return len(self.valuesL)

    def __getitem__(self, idx):
        sampleL  = self.valuesL[idx]
        sampleAB = self.valuesAB[idx]

        return sampleL, sampleAB