import os
import torch
import torch.nn as nn
import torch.optim as optim


from tqdm import tqdm
from torch.utils.data import DataLoader


from collections import defaultdict
from generator import ColorizationCNN, ColorizationCNN2
from discriminator import Discriminator
from dataset import ColorizationDataset
from parser import makeParser
from utils import undoTransform

from skimage import io
import numpy as np


args = makeParser().parse_args()

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
videoPaths = defaultdict(list)

for split in ["train", "test"]:
    for imgName in os.listdir(os.path.join(rootFolder, f"{split}_black")):
        imgBWPath    = os.path.join(rootFolder, f"{split}_black", imgName)
        imgColorPath = os.path.join(rootFolder, f"{split}_color", imgName)
        videoPaths[split].append([imgBWPath, imgColorPath])

videoPaths["train"] = videoPaths['train'][ : 50]


datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=1, shuffle=True)
datasetTest     = ColorizationDataset(videoPaths["test"])
dataloaderTest  = DataLoader(datasetTest, batch_size=1, shuffle=True)

device = args.device

# Instantiate model
generator     = ColorizationCNN().to(device)
discriminator = Discriminator().to(device)

if args.mode == "train":

    # Loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()  # For discriminator
    content_criterion = nn.L1Loss()      # For generator (L2 loss)

    # Optimizers
    lr_D=0.01
    lr_G=0.01
    alpha = 0
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G)

    schedulerG   = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[300,200,250], gamma=0.1)
    schedulerD   = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[30,200,250], gamma=0.1)


    numEpochs = 300
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        runningLossGenerator     = 0
        runningLossDiscriminator = 0
        runningLossAdversarial   = 0
        for i, (groundTruthL, groundTruthAB) in enumerate(tqdm(dataloaderTrain, desc=f"Epoch {currentEpoch+1}", leave=False, unit="batch")):
            groundTruthL   = groundTruthL.to(torch.float32).to(device)
            groundTruthAB  = groundTruthAB.to(torch.float32).to(device)
            
            generatedAB  = generator(groundTruthL)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_images = groundTruthAB
            fake_images = generatedAB.detach()

            predictionRealImages = discriminator(real_images)
            predictionFakeImages = discriminator(fake_images)

            real_labels = torch.empty_like(predictionRealImages).uniform_(0.85, 0.99)
            fake_labels = torch.zeros_like(predictionFakeImages)

            real_loss = adversarial_criterion(predictionRealImages, real_labels)
            fake_loss = adversarial_criterion(predictionFakeImages, fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2
            
            runningLossDiscriminator += discriminator_loss.item()
            discriminator_loss.backward()
            optimizer_D.step()
            
                
            # Train Generator
            optimizer_G.zero_grad()
            # Content loss
            mse_loss = content_criterion(generatedAB, groundTruthAB)
            
            # Adversarial loss
            adversarial_loss = adversarial_criterion(discriminator(fake_images), real_labels)
            if currentEpoch > 60:
                alpha = 0.7

            generator_loss   = mse_loss + alpha * adversarial_loss


            runningLossGenerator     += generator_loss.item()
            runningLossAdversarial   += adversarial_loss.item()
                
            generator_loss.backward()
            optimizer_G.step()


        schedulerG.step()
        schedulerD.step()
        print(f"\nEpoch {currentEpoch} finished. LR: {schedulerG.get_last_lr()} Generator Loss: {runningLossGenerator}, Discriminator Loss: {runningLossDiscriminator}, Adversarial Loss: {runningLossAdversarial}")

    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 5
        
        groundTruthL, groundTruthAB = datasetTest[imgIdx]
        groundTruthL = torch.from_numpy(groundTruthL).to(torch.float32).unsqueeze(0)
        groundTruthAB = torch.from_numpy(groundTruthAB).to(torch.float32)


        generatedAB  = generator(groundTruthL)
        
        # Remove batch channel
        generatedAB  = generatedAB.squeeze(0)
        # Remove batch channel
        groundTruthL = groundTruthL.squeeze(0).squeeze(0)

        generatedRGB    = undoTransform(groundTruthL, generatedAB)
        groundTruthRGB  = undoTransform(groundTruthL, groundTruthAB)
        
        io.imsave("./colorized.jpeg", generatedRGB)
        io.imsave("./groundtruth.jpeg", groundTruthRGB)