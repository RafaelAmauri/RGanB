import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from generator import ColorizationCNN
from preprocess import preprocessDataset


rootFolder = "/home/rafael/Estudos/Datasets/imagenette2-160"
componentsL, componentsAB = preprocessDataset(rootFolder, "./Preprocessed files")

Ltrain  = componentsL["train"]
ABtrain = componentsAB["train"]

Lval    = componentsL["val"]
ABval   = componentsAB["val"]


datasetTrain = ColorizationDataset(Ltrain, ABtrain)
dataloader   = DataLoader(datasetTrain, batch_size=32, shuffle=True)


device = "cuda"

# Instantiate model
generator     = ColorizationCNN().to(device)

# Binary cross-entropy loss
criterion = nn.MSELoss()

# Optimizers
lr=0.001
optimizer = optim.Adam(generator.parameters(), lr=lr)


for epoch in range(50):
    for i, (sampleL, sampleAB) in enumerate(dataloader):

        sampleL  = sampleL.float().to(device)
        sampleAB = sampleAB.float().to(device)

        generatedAB = generator(sampleL)

        optimizer.zero_grad()
        loss = criterion(generatedAB, sampleAB)

        loss.backward()
        optimizer.step()

        print(f"Epoch loss = {loss.item()}")