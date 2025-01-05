import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import reverseTransform
from collections import defaultdict
from generator import ColorizationCNN
from dataset import ColorizationDataset
from parser import makeParser


args = makeParser().parse_args()

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
videoPaths = defaultdict(list)

for split in ["train", "test"]:
    for imgName in os.listdir(os.path.join(rootFolder, f"{split}_color")):
        imgPath = os.path.join(rootFolder, f"{split}_color", imgName)
        videoPaths[split].append(imgPath)


datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=3, shuffle=True)

device = args.device

# Instantiate model
generator  = ColorizationCNN().to(device)


if args.mode == "train":
    # MSE loss
    criterion = nn.MSELoss()

    # Optimizers
    lr=0.001
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    numEpochs = 2
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        epochLoss = 0
        for i, (sampleL, sampleAB) in enumerate(tqdm(dataloaderTrain, desc=f"Epoch {currentEpoch+1}", leave=False, unit="batch")):

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



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 0
        
        groundTruthL, groundTruthAB = datasetTrain[0]
        # Add batch channel
        groundTruthL = groundTruthL.unsqueeze(0)

        generatedAB  = generator(groundTruthL)
        
        # Remove batch channel
        generatedAB  = generatedAB.squeeze(0)
        # Remove batch channel
        groundTruthL = groundTruthL.squeeze(0)

        
        img = reverseTransform(groundTruthL, generatedAB)

        from PIL import Image
        img = Image.fromarray(img)
        img.save("./a.png")