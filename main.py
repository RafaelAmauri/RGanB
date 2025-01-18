import os
import torch
import torch.nn as nn
import torch.optim as optim


from tqdm import tqdm
from torch.utils.data import DataLoader


from collections import defaultdict
from generator import ColorizationCNN
from discriminator import Discriminator, NLayerDiscriminator
from dataset import ColorizationDataset
from parser import makeParser
from utils import transform, undoTransform

from torch.amp import autocast, GradScaler
from skimage import io

from torchvision.models import vgg19
import torch.nn.functional as F


args = makeParser().parse_args()

rootFolder = "/home/rafael/Estudos/Datasets/image-colorization-dataset/data"
videoPaths = defaultdict(list)

for split in ["train", "test"]:
    for imgName in os.listdir(os.path.join(rootFolder, f"{split}_black")):
        imgBWPath    = os.path.join(rootFolder, f"{split}_black", imgName)
        imgColorPath = os.path.join(rootFolder, f"{split}_color", imgName)
        videoPaths[split].append([imgBWPath, imgColorPath])

videoPaths["train"] = videoPaths['train'][ : 20]


datasetTrain    = ColorizationDataset(videoPaths["train"])
dataloaderTrain = DataLoader(datasetTrain, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
datasetTest     = ColorizationDataset(videoPaths["test"])
dataloaderTest  = DataLoader(datasetTest,  batch_size=1, shuffle=True, num_workers=16, pin_memory=True)

device = args.device

# Instantiate model
generator     = ColorizationCNN().to(device)
discriminator = NLayerDiscriminator(3).to(device)

if args.mode == "train":

    # Loss functions
    adversarialCriterion = nn.BCEWithLogitsLoss()   # For discriminator
    l1Loss  = nn.L1Loss()

    # Optimizers
    lr_D=0.0001
    lr_G=0.001
    
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.5, 0.999))
    
    scalerG = GradScaler()
    scalerD = GradScaler()
    
    # This pretrained VGG model is used for perceptual loss. It identifies if the features of the
    # generated image are consistent with the features of the groundtruth image.
    vgg = vgg19(weights="VGG19_Weights.DEFAULT").features.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    numEpochs = 300
    nBatches  = max(1, len(datasetTrain) // dataloaderTrain.batch_size)

    for currentEpoch in tqdm(range(numEpochs), desc="Training Completed", unit="epoch"):
        runningLossGenerator     = 0
        runningLossDiscriminator = 0
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

            # Ligando o cálculo de gradiente para o discriminador
            for p in discriminator.parameters():
                p.requires_grad = True


            # Train Discriminator
            optimizerD.zero_grad()
            with autocast(device_type=device, dtype=torch.float16):
                # Discriminator learns to differentiate the entire groundtruth LAB image from the fake LAB image.
                # It receives all 3 channels, not just the A and B channels.
                predictionFakeImages = discriminator(generatedLAB.detach())
                arrayZeros           = torch.zeros_like(predictionFakeImages)
                fakeImagesLoss       = adversarialCriterion(predictionFakeImages, arrayZeros)

                predictionRealImages = discriminator(groundTruthLAB)
                arrayOnes            = torch.ones_like(predictionRealImages).uniform_(0.85, 0.95)
                realImagesLoss       = adversarialCriterion(predictionRealImages, arrayOnes)

                discriminatorLoss = (realImagesLoss + fakeImagesLoss) / 2
                runningLossDiscriminator += discriminatorLoss.item()


            scalerD.scale(discriminatorLoss).backward()
            scalerD.step(optimizerD)
            scalerD.update()
            

            # Train Generator
            for p in discriminator.parameters():
                p.requires_grad = False
        
            # Zeramos os gradientes do gerador
            optimizerG.zero_grad()
            
            with autocast(device_type=device, dtype=torch.float16):
                # L1 Loss is calculated for the A and B channels ONLY.
                l1_loss  = l1Loss(generatedAB, groundTruthAB)
                
                # Adversarial loss is calculated for the LAB image.
                if currentEpoch < 100:
                    generatorLoss = 100 * l1_loss
                else:
                    adversarialLoss = adversarialCriterion(discriminator(generatedLAB), torch.ones_like(predictionRealImages))
                    generatedRGB,   _ = undoTransform(generatedLAB.detach().cpu().squeeze(0))
                    groundTruthRGB, _ = undoTransform(groundTruthLAB.cpu().squeeze(0))

                    generatedRGB   = torch.from_numpy(generatedRGB).permute((2, 0, 1)).to(torch.float32).to(device)
                    groundTruthRGB = torch.from_numpy(groundTruthRGB).permute((2, 0, 1)).to(torch.float32).to(device)
                    perceptualLoss  = l1Loss(vgg(generatedRGB), vgg(groundTruthLAB))

                    generatorLoss   = adversarialLoss + 80 * l1_loss + perceptualLoss

                runningLossGenerator     += generatorLoss.item()
                
            # Scale loss and backpropagate for generator
            scalerG.scale(generatorLoss).backward()
            scalerG.step(optimizerG)
            scalerG.update()


        runningLossGenerator     = runningLossGenerator     / nBatches
        runningLossDiscriminator = runningLossDiscriminator / nBatches

        print(f"\nEpoch {currentEpoch} finished. Generator Loss (mean/batch): {runningLossGenerator}, Discriminator Loss (mean/batch): {runningLossDiscriminator}")


    torch.save(generator.state_dict(), "Generator.pth")
    print("Trained model saved to Generator.pth")



elif args.mode == "test":
    generator.load_state_dict(torch.load("Generator.pth", weights_only=True))
    generator = generator.to(device)
    
    with torch.no_grad():
        imgIdx = 11
        
        groundTruthL, groundTruthAB, groundTruthLAB = datasetTrain[imgIdx]

        groundTruthL   = groundTruthL.to(torch.float32).unsqueeze(0)
        groundTruthLAB = groundTruthLAB.to(torch.float32)

        generatedAB  = generator(groundTruthL)
        generatedLAB = torch.cat((groundTruthL, generatedAB), dim=1)
        
        # Remove batch channel
        generatedLAB  = generatedLAB.squeeze(0)
        # Remove batch channel
        groundTruthL = groundTruthL.squeeze(0).squeeze(0)

        generatedRGB, _    = undoTransform(generatedLAB)
        groundTruthRGB, groundTruthGray  = undoTransform(groundTruthLAB)
        
        io.imsave("./colorized.jpeg", generatedRGB)
        io.imsave("./groundtruth.jpeg", groundTruthRGB)
        io.imsave("./grayscale.jpeg", groundTruthGray)