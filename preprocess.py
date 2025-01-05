import os
import cv2
import torch
import numpy as np

from tqdm import tqdm


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
            componentsL[split]  = []
            componentsAB[split] = []

            colorImgNames = os.listdir(os.path.join(datasetPath, f"{split}_color"))
            for colorImgName in tqdm(colorImgNames, desc=f"Processing {split} images", unit="image"):
                colorImgPath         = os.path.join(datasetPath, f"{split}_color", colorImgName)
                
                colorImg = cv2.imread(colorImgPath)
                colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                colorImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2LAB)  # Convert from RGB to LAB
                
                colorImg = np.array(colorImg)

                l = colorImg[:, :, 0]  # The l component is the first channel
                a = colorImg[:, :, 1]  # The a component is the second channel
                b = colorImg[:, :, 2]  # The b component is the third channel

                l  = torch.tensor(l, dtype=torch.float32).unsqueeze(0) # Shape (1, H, W)
                ab = torch.tensor(np.stack((a, b), axis=-1), dtype=torch.float32).permute(2, 0, 1)  # Shape (2, H, W)

                l  = l / 255.
                ab = ab / 255.

                componentsL[split].append(l)
                componentsAB[split].append(ab)


            componentsL[split]  = np.array(componentsL[split])
            componentsAB[split] = np.array(componentsAB[split])

            np.save(os.path.join(outputPath, f'{split}_componentsL.npy'), componentsL[split])
            print(f"Saved L components in {os.path.join(outputPath, f'{split}_componentsL.npy')}")
            
            np.save(os.path.join(outputPath, f'{split}_componentsAB.npy'), componentsAB[split])
            print(f"Saved A,B components in {os.path.join(outputPath, f'{split}_componentsAB.npy')}")

    
    return componentsL, componentsAB