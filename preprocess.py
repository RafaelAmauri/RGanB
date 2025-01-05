import os
import numpy as np

import torchvision

from tqdm import tqdm
from skimage.color import rgb2lab
from utils import transform


def preprocessDataset(datasetPath, outputPath):
    """
    Opens the dataset folder and converts the images to LAB. 
    Then, split then the L, A and B components from the image.

    Lastly, save the L and AB components into separate .npy file.

    Args:
        datasetPath (str): Path to the Imagenette dataset
        outputPath  (str): Where to save the .npy files.
    """

    componentsL  = {}
    componentsAB = {}

    if os.path.exists(outputPath):
        print(f"{outputPath} already exists! Loading already preprocessed annotations...")
        
        for split in ["train", "test"]:
            componentsL[split]  = np.load(os.path.join(outputPath, f"{split}_componentsL.npy"))
            componentsAB[split] = np.load(os.path.join(outputPath, f"{split}_componentsAB.npy"))

    else:
        os.mkdir(outputPath)
        
        for split in ["train", "test"]:
            addedImgs = 0
            componentsL[split]  = []
            componentsAB[split] = []

            colorImgNames = os.listdir(os.path.join(datasetPath, f"{split}_color"))
            for colorImgName in tqdm(colorImgNames, desc=f"Processing {split} images", unit="image"):
                if addedImgs >= 10:
                    break
                colorImgPath  = os.path.join(datasetPath, f"{split}_color", colorImgName)
                
                colorImg = torchvision.io.read_image(colorImgPath)
                
                l, ab = transform(colorImg)
                
                componentsL[split].append(l)
                componentsAB[split].append(ab)

                addedImgs += 1


            componentsL[split]  = np.array(componentsL[split])
            componentsAB[split] = np.array(componentsAB[split])

            np.save(os.path.join(outputPath, f'{split}_componentsL.npy'), componentsL[split])
            print(f"Saved L components in {os.path.join(outputPath, f'{split}_componentsL.npy')}")
            
            np.save(os.path.join(outputPath, f'{split}_componentsAB.npy'), componentsAB[split])
            print(f"Saved A,B components in {os.path.join(outputPath, f'{split}_componentsAB.npy')}")

    
    return componentsL, componentsAB