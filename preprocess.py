import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2lab


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
        
        for split in ["train", "val"]:
            componentsL[split]  = np.load(os.path.join(outputPath, f"{split}_componentsL.npy"))
            componentsAB[split] = np.load(os.path.join(outputPath, f"{split}_componentsAB.npy"))

    else:
        os.mkdir(outputPath)
        
        for split in ["train", "val"]:
            componentsL[split]  = []
            componentsAB[split] = []

            for imageFolder in os.listdir(os.path.join(datasetPath, split)):
                imagePaths = os.listdir(os.path.join(datasetPath, split, imageFolder))
                
                for imgPath in tqdm(imagePaths, desc=f"Processing {split} - {imageFolder}", unit="image"):
                    imgPath = os.path.join(datasetPath, split, imageFolder, imgPath)
                    
                    img = Image.open(imgPath)
                    img = img.resize((2, 2)) # Resize to 255x255 and convert to LAB
                    imgRGB = img.copy().convert("RGB")
                    img = img.convert("LAB")
                    
                    # CADA UMA DAS CONVERSOES PARA LAB ESTA GERANDO RESULTADOS DIFERENTES :(
                    img_rgb_np = np.array(imgRGB)
                    img_np = np.array(img)
                    img_np2 = np.array(rgb2lab(imgRGB))
                    print(img_rgb_np)
                    print(img_np)
                    print(img_np2)

                    print(img_np == img_np2)
                    raise Exception

                    l = img_np[:, :, 0]  # The l component is the first channel
                    a = img_np[:, :, 1]  # The a component is the second channel
                    b = img_np[:, :, 2]  # The b component is the third channel

                    l  = torch.tensor(l, dtype=torch.float32).unsqueeze(0) # Shape (1, H, W)
                    ab = torch.tensor(np.stack((a, b), axis=-1), dtype=torch.float32).permute(2, 0, 1)  # Shape (2, H, W)

                    l  = l / 100.
                    ab = ab / 128.

                    print(l)
                    print(ab)

                    componentsL[split].append(l)
                    componentsAB[split].append(ab)


            componentsL[split]  = np.array(componentsL[split])
            componentsAB[split] = np.array(componentsAB[split])

            np.save(os.path.join(outputPath, f'{split}_componentsL.npy'), componentsL[split])
            print(f"Saved L components in {os.path.join(outputPath, f'{split}_componentsL.npy')}")
            
            np.save(os.path.join(outputPath, f'{split}_componentsAB.npy'), componentsAB[split])
            print(f"Saved A,B components in {os.path.join(outputPath, f'{split}_componentsAB.npy')}")

    
    return componentsL, componentsAB