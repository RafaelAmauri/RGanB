import skimage
import numpy as np
from skimage import io, color
from skimage.transform import resize


def undoTransform(imgColorLAB):
    imgColorL, imgColorA, imgColorB = imgColorLAB

    imgColorL = imgColorL * 100
    imgColorA = (imgColorA * 256 ) - 128
    imgColorB = (imgColorB * 256)  - 128

    imgColorLAB = np.stack((imgColorL, imgColorA, imgColorB), axis=0)
    imgColorLAB = np.transpose(imgColorLAB, (1, 2, 0)) # Convert from [C, H, W] to [H, W, C]
    

    imgRGB  = color.lab2rgb(imgColorLAB)
    imgRGB  = skimage.util.img_as_ubyte(imgRGB)
    
    return imgRGB



def transform(imgColor):

    imgColor = resize(imgColor, (256,256))

    # Convert to LAB colorspace
    imgColor = color.rgb2lab(imgColor)

    # Convert from [H, W, C] to [C, H, W] for pytorch to work properly
    imgColorLAB = np.transpose(imgColor, (2, 0, 1))

    imgColorLNormalized = imgColorLAB[0] / 100
    imgColorANormalized = (imgColorLAB[1] + 128) / 256 # The 'a' component is the second channel
    imgColorBNormalized = (imgColorLAB[2] + 128) / 256 # The 'b' component is the third channel

    imgColorLABNormalized = np.stack((imgColorLNormalized, imgColorANormalized, imgColorBNormalized), axis=0)
    imgColorABNormalized  = np.stack((imgColorANormalized, imgColorBNormalized), axis=0)
    imgColorLNormalized   = np.expand_dims(imgColorLNormalized, axis=0)

    return imgColorLNormalized, imgColorABNormalized, imgColorLABNormalized