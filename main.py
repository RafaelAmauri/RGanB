import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import reverseTransform, undo_transform
from collections import defaultdict
from generator import ColorizationCNN, ColorizationCNN2
from discriminator import Discriminator
from dataset import ColorizationDataset
from parser import makeParser

import numpy as np


args = makeParser().parse_args()

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
videoPaths = defaultdict(list)

for split in ["train", "test"]:
    for imgName in os.listdir(os.path.join(rootFolder, f"{split}_color")):
        imgPath = os.path.join(rootFolder, f"{split}_color", imgName)
        videoPaths[split].append(imgPath)


videoPaths["train"] = videoPaths["train"][:500]

datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=24, shuffle=True)
datasetTest     = ColorizationDataset(videoPaths["test"])
dataloaderTest  = DataLoader(datasetTest, batch_size=24, shuffle=True)

device = args.device

# Instantiate model
generator     = ColorizationCNN().to(device)
discriminator = Discriminator().to(device)


if args.mode == "train":

    # Loss functions
    adversarial_criterion = nn.BCELoss()  # For discriminator
    content_criterion = nn.MSELoss()      # For generator (L2 loss)

    # Optimizers
    lr=0.001

    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)


    numEpochs = 30
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        runningLossGenerator = 0
        runningLossDiscriminator = 0
        runningLossAdversarial = 0
        for i, (sampleL, sampleAB) in enumerate(tqdm(dataloaderTrain, desc=f"Epoch {currentEpoch+1}", leave=False, unit="batch")):

            sampleL  = sampleL.float().to(device)
            sampleAB = sampleAB.float().to(device)

            generatedAB  = generator(sampleL)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_images = torch.cat([sampleL, sampleAB], dim=1)
            fake_images = torch.cat([sampleL, generatedAB.detach()], dim=1)

            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(fake_images.size(0), 1).to(device)

            real_loss = adversarial_criterion(discriminator(real_images), real_labels)
            fake_loss = adversarial_criterion(discriminator(fake_images), fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2
            
            discriminator_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()

             # Content loss
            mse_loss = content_criterion(generatedAB, sampleAB)
            
            # Adversarial loss
            real_loss = adversarial_criterion(discriminator(real_images), fake_labels)
            fake_loss = adversarial_criterion(discriminator(fake_images), real_labels)
            adversarial_loss = (real_loss + fake_loss) / 2
            generator_loss = 0.01 * adversarial_loss

            generator_loss.backward()
            optimizer_G.step()

            runningLossGenerator += generator_loss.item()
            runningLossDiscriminator += discriminator_loss.item()
            runningLossAdversarial += adversarial_loss.item()

        print(f"\nEpoch {currentEpoch} finished. Generator Loss: {runningLossGenerator}, Discriminator Loss: {runningLossDiscriminator}, Adversarial Loss: {runningLossAdversarial}")

    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 0
        
        groundTruthL, groundTruthAB = datasetTrain[imgIdx]
        # Add batch channel
        groundTruthL = groundTruthL.unsqueeze(0)

        generatedAB  = generator(groundTruthL)
        
        # Remove batch channel
        generatedAB  = generatedAB.squeeze(0)
        # Remove batch channel
        groundTruthL = groundTruthL.squeeze(0)

        from PIL import Image
        
        img = reverseTransform(groundTruthL, generatedAB)
        img = Image.fromarray(img)
        img.save("./colorized.png")

        img = reverseTransform(groundTruthL, groundTruthAB)
        img = Image.fromarray(img)
        img.save("./groundtruth.png")

        
        img = Image.fromarray(undo_transform(groundTruthL))
        img.save("./black_and_white.png")