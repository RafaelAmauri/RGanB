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

from torch.amp import autocast, GradScaler
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

videoPaths["train"] = videoPaths['train'][ : 300]


datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
datasetTest     = ColorizationDataset(videoPaths["test"])
dataloaderTest  = DataLoader(datasetTest, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

device = args.device

# Instantiate model
generator     = ColorizationCNN().to(device)
discriminator = Discriminator().to(device)

if args.mode == "train":

    # Loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()   # For discriminator
    mseLoss = nn.MSELoss()                           # For generator (L2 loss)
    l1Loss  = nn.L1Loss()

    # Optimizers
    lr_D=0.001
    lr_G=0.001
    alpha=0.001
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G)

    schedulerG   = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[500], gamma=0.1)
    schedulerD   = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[500], gamma=0.1)
    
    scaler_G = GradScaler()
    scaler_D = GradScaler()

    numEpochs = 150
    nBatches  = max(1, len(datasetTrain) // dataloaderTrain.batch_size)
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        runningLossGenerator     = 0
        runningLossDiscriminator = 0
        runningLossAdversarial   = 0
        for (groundTruthL, groundTruthAB, groundTruthLAB) in tqdm(dataloaderTrain, desc=f"Epoch {currentEpoch}", leave=False, unit="batch"):
            groundTruthL    = groundTruthL.to(torch.float32).to(device)
            groundTruthLAB  = groundTruthLAB.to(torch.float32).to(device)
            groundTruthAB   = groundTruthAB.to(torch.float32).to(device)


            # The generator only generates the A and B channels. The L channel is later added to the tensor to form the LAB image.
            # The idea is to avoid the generator messing with the L channel in the convolutions. Since the generator doesn't mess with the L channel, 
            # it will focus its effort only on adjusting the weights for the A and B channels.
            with autocast(device_type=device, dtype=torch.float16):
                generatedAB  = generator(groundTruthL)

            generatedLAB = torch.cat((groundTruthL, generatedAB), dim=1)

            # Train Discriminator
            with autocast(device_type=device, dtype=torch.float16):
                # Discriminator learns to differentiate the entire groundtruth LAB image from the fake LAB image.
                # It receives all 3 channels, not just the A and B channels.
                real_images = groundTruthLAB
                fake_images = generatedLAB

                predictionRealImages = discriminator(real_images)
                predictionFakeImages = discriminator(fake_images)

                real_labels = torch.ones_like(predictionRealImages)
                fake_labels = torch.zeros_like(predictionFakeImages)

                real_loss = adversarial_criterion(predictionRealImages, real_labels)
                fake_loss = adversarial_criterion(predictionFakeImages, fake_labels)
                discriminator_loss = real_loss + fake_loss
                
                runningLossDiscriminator += discriminator_loss.item()

            scaler_D.scale(discriminator_loss).backward(retain_graph=True)
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
                
            # Train Generator
            optimizer_G.zero_grad()
            with autocast(device_type=device, dtype=torch.float16):
                # MSE Loss is calculated for the A and B channels ONLY.
                mse_loss = mseLoss(generatedAB, groundTruthAB)
                l1_loss  = l1Loss(generatedAB, groundTruthAB)
                # Adversarial loss is calculated for the LAB image.
                adversarial_loss = adversarial_criterion(discriminator(fake_images), real_labels)
                
                generator_loss = adversarial_loss

                runningLossGenerator     += generator_loss.item()
                runningLossAdversarial   += adversarial_loss.item()
                
            # Scale loss and backpropagate for generator
            scaler_G.scale(generator_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()


        runningLossGenerator     = runningLossGenerator     / nBatches
        runningLossDiscriminator = runningLossDiscriminator / nBatches
        runningLossAdversarial   = runningLossAdversarial   / nBatches
        schedulerG.step()
        schedulerD.step()

        print(f"\nEpoch {currentEpoch} finished. LR: {schedulerG.get_last_lr()} Generator Loss (mean/batch): {runningLossGenerator}, Discriminator Loss (mean/batch): {runningLossDiscriminator}, Adversarial Loss (mean/batch): {runningLossAdversarial}")

    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():
        imgIdx = 0
        
        groundTruthL, groundTruthAB, groundTruthLAB = datasetTrain[imgIdx]

        groundTruthL   = torch.from_numpy(groundTruthL).to(torch.float32).unsqueeze(0)
        groundTruthLAB = torch.from_numpy(groundTruthLAB).to(torch.float32)

        generatedAB  = generator(groundTruthL)
        generatedLAB = torch.cat((groundTruthL, generatedAB), dim=1)
        
        # Remove batch channel
        generatedLAB  = generatedLAB.squeeze(0)
        # Remove batch channel
        groundTruthL = groundTruthL.squeeze(0).squeeze(0)

        generatedRGB    = undoTransform(generatedLAB)
        groundTruthRGB  = undoTransform(groundTruthLAB)
        
        io.imsave("./colorized.jpeg", generatedRGB)
        io.imsave("./groundtruth.jpeg", groundTruthRGB)