import torch
import skimage

from skimage import io, color
from skimage.transform import resize


def undoTransform(imgColorLAB):
    imgColorL, imgColorA, imgColorB = imgColorLAB

    imgColorL = imgColorL * 100
    imgColorA = (imgColorA * 256 ) - 128
    imgColorB = (imgColorB * 256)  - 128

    imgColorLAB = torch.stack((imgColorL, imgColorA, imgColorB), axis=0)
    imgColorLAB = imgColorLAB.permute((1, 2, 0)) # Convert from [C, H, W] to [H, W, C]

    imgRGB  = color.lab2rgb(imgColorLAB)
    imgRGB  = skimage.util.img_as_ubyte(imgRGB)
    imgGray = color.rgb2gray(imgRGB)
    imgGray = skimage.util.img_as_ubyte(imgGray)

    return imgRGB, imgGray



def transform(imgColor):

    imgColor = resize(imgColor, (256,256))

    # Convert to LAB colorspace
    imgColor = color.rgb2lab(imgColor)

    # Convert from [H, W, C] to [C, H, W] for pytorch to work properly
    imgColorLAB = torch.from_numpy(imgColor).permute((2, 0, 1))

    imgColorLNormalized = imgColorLAB[0] / 100
    imgColorANormalized = (imgColorLAB[1] + 128) / 256 # The 'a' component is the second channel
    imgColorBNormalized = (imgColorLAB[2] + 128) / 256 # The 'b' component is the third channel

    imgColorLABNormalized = torch.stack((imgColorLNormalized, imgColorANormalized, imgColorBNormalized), axis=0)
    imgColorABNormalized  = torch.stack((imgColorANormalized, imgColorBNormalized), axis=0)
    imgColorLNormalized   = imgColorLNormalized.unsqueeze(0)

    return imgColorLNormalized, imgColorABNormalized, imgColorLABNormalized