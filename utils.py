import skimage
import numpy as np
from skimage import io, color
from skimage.transform import resize


def undoTransform(imgColorL, imgColorAB):
    imgColorA, imgColorB = imgColorAB

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
    imgColor = np.transpose(imgColor, (2, 0, 1))

    imgColorL = imgColor[0]
    imgColorL = np.expand_dims(imgColorL, axis=0)
    imgColorA = imgColor[1] # The 'a' component is the second channel
    imgColorB = imgColor[2] # The 'b' component is the third channel

    imgColorAB = np.stack((imgColorA, imgColorB), axis=0)

    return imgColorL, imgColorAB