import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from generator import ColorizationCNN
from preprocess import preprocessDataset

import cv2
import matplotlib.pyplot as plt


mode = "TEST"

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
componentsL, componentsAB = preprocessDataset(rootFolder, "./Preprocessed files")

Ltrain   = componentsL["train"]
ABtrain  = componentsAB["train"]

Ltest    = componentsL["test"]
ABtest   = componentsAB["test"]


datasetTrain = ColorizationDataset(Ltrain, ABtrain)
dataloader   = DataLoader(datasetTrain, batch_size=48, shuffle=True)

device = "cuda"

# Instantiate model
generator  = ColorizationCNN().to(device)

if mode == "TRAIN":
    # MSE loss
    criterion = nn.MSELoss()

    # Optimizers
    lr=0.001
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    numEpochs = 300
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        epochLoss = 0
        for i, (sampleL, sampleAB) in enumerate(tqdm(dataloader, desc=f"Epoch {currentEpoch+1}", leave=False, unit="batch")):

            sampleL  = sampleL.float().to(device)
            sampleAB = sampleAB.float().to(device)

            generatedAB = generator(sampleL)

            optimizer.zero_grad()

            loss = criterion(generatedAB, sampleAB)

            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        print(f"\nEpoch {currentEpoch} finished. Loss: {epochLoss}")

    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")

elif mode == "TEST":
    
    generator.load_state_dict(torch.load("cnn_6th_2000_full.pt")["model_state_dict"])
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 0

        groundTruthL  = torch.tensor(Ltest[imgIdx],  dtype=torch.float32).unsqueeze(0).to(device)
        groundTruthAB = torch.tensor(ABtest[imgIdx], dtype=torch.float32).to(device)


        generatedAB = generator(groundTruthL)

        groundTruthL = groundTruthL.to("cpu").squeeze(0).squeeze(0)
        groundTruthL = groundTruthL * 255.

        generatedAB = generatedAB.to("cpu")
        generatedAB = generatedAB.detach().squeeze(0)
        generatedAB = generatedAB * 255.

        generatedA, generatedB = generatedAB
        
        LAB_img = np.stack([groundTruthL, generatedA, generatedB], axis=-1).astype(np.uint8)
        RGB_img = cv2.cvtColor(LAB_img, cv2.COLOR_LAB2RGB)

        plt.imshow(RGB_img)
        plt.title("Converted RGB Image")
        plt.axis('off')
        plt.show()