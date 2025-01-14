import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler


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
    for imgName in os.listdir(os.path.join(rootFolder, f"{split}_black")):
        imgBWPath    = os.path.join(rootFolder, f"{split}_black", imgName)
        imgColorPath = os.path.join(rootFolder, f"{split}_color", imgName)
        videoPaths[split].append([imgBWPath, imgColorPath])

videoPaths["train"] = videoPaths['train'][ : 100]
useAmp = True


datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True)
datasetTest     = ColorizationDataset(videoPaths["test"])
dataloaderTest  = DataLoader(datasetTest, batch_size=32, shuffle=True)

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
    alpha = 0.001
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G)

    schedulerG   = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[130,200,250], gamma=0.1)
    schedulerD   = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[30,200,250], gamma=0.1)

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    numEpochs = 200
    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        runningLossGenerator     = 0
        runningLossDiscriminator = 0
        runningLossAdversarial   = 0
        for i, (sampleBW, sampleRGB) in enumerate(tqdm(dataloaderTrain, desc=f"Epoch {currentEpoch+1}", leave=False, unit="batch")):
            sampleBW  = sampleBW.to(device)
            sampleRGB = sampleRGB.to(device)

            with autocast(device_type=device, dtype=torch.float16):
                generatedRGB  = generator(sampleBW)
                

            # Train Discriminator
            optimizer_D.zero_grad()
            with autocast(device_type=device, dtype=torch.float16):
                real_images = sampleRGB
                fake_images = generatedRGB.detach()

                predictionRealImages = discriminator(real_images)
                predictionFakeImages = discriminator(fake_images)

                real_labels = torch.empty_like(predictionRealImages).uniform_(0.95, 0.99)
                fake_labels = torch.zeros_like(predictionFakeImages)

                real_loss = adversarial_criterion(predictionRealImages, real_labels)
                fake_loss = adversarial_criterion(predictionFakeImages, fake_labels)
                discriminator_loss = (real_loss + fake_loss) / 2
                
                runningLossDiscriminator += discriminator_loss.item()

            # Scale loss and backpropagate for discriminator
            scaler_D.scale(discriminator_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()


            # Train Generator
            optimizer_G.zero_grad()
            with autocast(device_type=device, dtype=torch.float16):
                # Content loss
                mse_loss = content_criterion(generatedRGB, sampleRGB)
                
                # Adversarial loss
                adversarial_loss = adversarial_criterion(discriminator(fake_images), real_labels)
                if currentEpoch > 70:
                    alpha = 0.1
                    
                generator_loss   = mse_loss + alpha * adversarial_loss

                runningLossGenerator     += generator_loss.item()
                runningLossAdversarial   += adversarial_loss.item()

            # Scale loss and backpropagate for generator
            scaler_G.scale(generator_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()




        schedulerG.step()
        schedulerD.step()
        print(f"\nEpoch {currentEpoch} finished. LR: {schedulerG.get_last_lr()} Generator Loss: {runningLossGenerator}, Discriminator Loss: {runningLossDiscriminator}, Adversarial Loss: {runningLossAdversarial}")

    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():

        imgIdx = 1
        
        groundTruthBW, groundTruthRGB = datasetTrain[imgIdx]
        # Add batch channel
        groundTruthBW = groundTruthBW.unsqueeze(0)

        generatedRGB  = generator(groundTruthBW)
        
        # Remove batch channel
        generatedRGB  = generatedRGB.squeeze(0)
        # Remove batch channel
        groundTruthBW = groundTruthBW.squeeze(0)


        groundTruthRGB = (groundTruthRGB * 255).to(torch.uint8)
        generatedRGB   = (generatedRGB * 255).to(torch.uint8)
        groundTruthBW  = (groundTruthBW * 255).to(torch.uint8)
        
        content_criterion = nn.MSELoss()
        mse_loss = content_criterion(generatedRGB.to(torch.float), groundTruthRGB.to(torch.float))

        import torchvision 
        torchvision.io.write_jpeg(generatedRGB, "./colorized.jpeg")
        torchvision.io.write_jpeg(groundTruthRGB, "./groundtruth.jpeg")
        torchvision.io.write_jpeg(groundTruthBW, "./black_and_white.jpeg")
        