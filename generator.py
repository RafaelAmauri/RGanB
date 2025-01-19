import torch
import torch.nn as nn


class ColorizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.relu      = nn.ReLU()
        self.tanh      = nn.Tanh()
        self.sigmoid   = nn.Sigmoid()
        

        self.groupNorm2    = nn.GroupNorm(num_groups=1, num_channels=2)
        self.batchNorm3    = nn.BatchNorm2d(num_features=3)
        self.batchNorm64   = nn.BatchNorm2d(num_features=64)
        self.batchNorm128  = nn.BatchNorm2d(num_features=128)
        self.batchNorm256  = nn.BatchNorm2d(num_features=256)
        self.batchNorm512  = nn.BatchNorm2d(num_features=512)
        self.batchNorm1024 = nn.BatchNorm2d(num_features=1024)

        
        # These will downsize the image
        self.conv1_1    = nn.Conv2d(in_channels=1,  out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2    = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

        
        self.conv2      = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        

        self.conv3      = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        

        self.conv4      = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        
        
        self.conv5      = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)


        self.conv6      = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
       

        self.conv7      = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        
        
        # These will upsize the image.
        self.conv8      = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upconv1    = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.conv9      = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upconv2    = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv10     = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upconv3    = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv11    = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upconv4   = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv12    = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.upconv5   = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv13    = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upconv6   = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv14    = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upconv7   = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv15    = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0)


    def forward(self, LComponent):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        
        # Layer 1_1: [256,256,1] -> [256,256,64]
        layer1_1 = self.conv1_1(LComponent)
        
        # Layer 1_2: [256,256,64] -> [128,128,64]
        layer1_2 = self.conv1_2(layer1_1)
        layer1_2 = self.batchNorm64(layer1_2)
        layer1_2 = self.relu(layer1_2)

        # Layer 2: [128,128,64] -> [64,64,128]
        layer2 = self.conv2(layer1_2)
        layer2 = self.batchNorm128(layer2)
        layer2 = self.relu(layer2)

        # Later 3: [64,64,128] -> [32,32,256]
        layer3 = self.conv3(layer2)
        layer3 = self.batchNorm256(layer3)
        layer3 = self.relu(layer3)

        # Later 4: [32,32,256] -> [16,16,512]
        layer4 = self.conv4(layer3)
        layer4 = self.batchNorm512(layer4)
        layer4 = self.relu(layer4)

        # Layer 5: [16,16,512] -> [8,8,512]
        layer5 = self.conv5(layer4)
        layer5 = self.batchNorm512(layer5)
        layer5 = self.relu(layer5)

        # Layer 6: [8,8,512] -> [4,4,512]
        layer6 = self.conv6(layer5)
        layer6 = self.batchNorm512(layer6)
        layer6 = self.relu(layer6)

        # Layer 7: [4,4,512] -> [2,2,512]
        layer7 = self.conv7(layer6)
        layer7 = self.batchNorm512(layer7)
        layer7 = self.relu(layer7)

        # Layer 8: [2,2,512] -> [4,4,512]
        layer8 = self.conv8(layer7)
        layer8 = self.upconv1(layer8)
        layer8 = self.batchNorm512(layer8)
        layer8 = self.relu(layer8)


        # Layer 9: [4,4,512] -> [8,8,512]
        # Add residuals from layer 6
        layer9 = torch.cat((layer8, layer6), dim=1)
        layer9 = self.conv9(layer9)
        layer9 = self.upconv2(layer9)
        layer9 = self.batchNorm512(layer9)
        layer9 = self.relu(layer9)


        # Layer 10: [8,8,512] -> [16,16,512]
        # Add residuals from layer 5
        layer10 = torch.cat((layer9, layer5), dim=1)
        layer10 = self.conv10(layer10)
        layer10 = self.upconv3(layer10)
        layer10 = self.batchNorm512(layer10)
        layer10 = self.relu(layer10)


        # Layer 11: [16,16,512] -> [32,32,256]
        # Add residuals from layer 4
        layer11 = torch.cat((layer10, layer4), dim=1)
        layer11 = self.conv11(layer11)
        layer11 = self.upconv4(layer11)
        layer11 = self.batchNorm256(layer11)
        layer11 = self.relu(layer11)


        # Layer 12: [32,32,256] -> [64,64,128]
        # Add residuals from Layer 3
        layer12 = torch.cat((layer11, layer3), dim=1)
        layer12 = self.conv12(layer12)
        layer12 = self.upconv5(layer12)
        layer12 = self.batchNorm128(layer12)
        layer12 = self.relu(layer12)


        # Layer 13: [64,64,128] -> [128,128,64]
        # Add residuals from Layer 2
        layer13 = torch.cat((layer12, layer2), dim=1)
        layer13 = self.conv13(layer13)
        layer13 = self.upconv6(layer13)
        layer13 = self.batchNorm64(layer13)
        layer13 = self.relu(layer13)


        # Layer 14: [128,128,64] -> [256,256,64]
        # Add residuals from Layer 1_2
        layer14 = torch.cat((layer13, layer1_2), dim=1)
        layer14 = self.conv14(layer14)
        layer14 = self.upconv7(layer14)
        layer14 = self.batchNorm64(layer14)
        layer14 = self.relu(layer14)


        # Layer 15: [256,256,64] -> [256,256,3]
        # Add residuals from Layer 1_1
        layer15 = torch.cat((layer14, layer1_1), dim=1)
        layer15 = self.conv15(layer15)
        layer15 = self.sigmoid(layer15)

        return layer15