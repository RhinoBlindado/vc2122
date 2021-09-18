#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 [CASTELLANO]
 
Practica 0
Asignatura: Vision por Computador
Autor: Valentino Lugli (Github: @RhinoBlindado)
Septiembre 2021

    ----
    
[ENGLISH]

Practice 0
Course: Computer Vision
Author: Valentino Lugli (Github: @RhinoBlindado)
September 2021

"""

# LIBRARIES

#   Using Matplotlib to show images
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#   Using OpenCV for everything else related to images.
import cv2 as cv

#   
import numpy as np

#

# FUNCTIONS
def leeImagen(filename, flagColor):
    """
    Read an image from file.
    
    Parameters
    ----------
    filename : String
        Path to a valid image file.
    flagColor : Boolean
        Value indicating to read the image with RBG (True) or Grayscale (False).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return cv.imread(filename, int(flagColor))


def pintaI(im, title=None):
    """
    Print an arbitrary real number matrix

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    # Make each image appear on its own window with optional title.
    plt.figure(title)
    
    # Check if image is grayscale or RGB
    if len(im.shape) == 2:
        # Colormap it to grey and autonormalize values to between 0 and 1.
        plt.imshow(im, cmap='gray', norm=clr.Normalize())
    else:
        imAux = (im - np.min(im)) / (np.max(im) - np.min(im))
        plt.imshow(imAux[:,:,::-1])


def pintaIM(vim):
    minHeight = min(i.shape[0] for i in vim)
    
    for i in vim:
        pass

# MAIN
def main():
    """
    Main function from which all the other functions are to be called.

    Returns
    -------
    None.

    """

    #   Loading the test images.
    srcOrapple = "./images/orapple.jpg"
    srcMessi = "./images/messi.jpg"
    srcLogo = "./images/logoOpenCV.jpg"
    srcDave = "./images/dave.jpg"

    
    # Task 1: Read an image
    #########
    
    #   Reading image as greyscale.
    imageOrappleGrey = leeImagen(srcOrapple, False)
    #   Now with colors.
    imageOrappleColor = leeImagen(srcOrapple, True)
    

    # Task 2: Visualize an arbitrary real number matrix
    ##########
    
    #   Generating random matrices with different ranges
    greyScaleMatrix = np.random.default_rng().uniform(-10.0, 220.0, (8,8))
    colorMatrix = np.random.default_rng().uniform(-10.0, 16.0, (8,8,3))
    
    #   Showing the Orapples along with the matrices
    new = cv.vconcat([imageOrappleColor, imageOrappleColor])
    pintaI(imageOrappleGrey)
    pintaI(new)
    pintaI(greyScaleMatrix)
    pintaI(colorMatrix)
    
    # Task 3: Concatenate multiple images into one
    ##########
    
    #   Reading the remaining images
    imageVector=[]
    imageVector.append(leeImagen(srcMessi, True))
    imageVector.append(leeImagen(srcLogo, True))
    imageVector.append(leeImagen(srcDave, True))
    
    pintaIM(imageVector)


# Setting up the script enviroment
if __name__ == "__main__":
    main()