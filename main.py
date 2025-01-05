import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import reverseTransform
from generator import ColorizationCNN
from dataset import ColorizationDataset
from preprocess import preprocessDataset


mode = "TEST"

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
componentsL, componentsAB = preprocessDataset(rootFolder, "./Preprocessed files")

Ltrain   = componentsL["train"]
ABtrain  = componentsAB["train"]

Ltest    = componentsL["test"]
ABtest   = componentsAB["test"]


datasetTrain = ColorizationDataset(Ltrain, ABtrain)
dataloader   = DataLoader(datasetTrain, batch_size=48, shuffle=True)

device = "cpu"

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
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 0

        groundTruthL  = torch.tensor(Ltest[imgIdx],  dtype=torch.float32).unsqueeze(0).to(device)
        groundTruthAB = torch.tensor(ABtest[imgIdx], dtype=torch.float32).to(device)


        generatedAB  = generator(groundTruthL)
        generatedAB  = generatedAB.squeeze(0)
        
        groundTruthL = groundTruthL.squeeze(0)
        
        img = reverseTransform(groundTruthL, generatedAB)

        from PIL import Image
        img = Image.fromarray(img)
        img.save("./a.png")