#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 0
    Asignatura: Vision por Computador
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Septiembre 2021
    
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

#   Using Numpy to manipulate images
import numpy as np

# FUNCTIONS
def leeImagen(filename, flagColor):
    """
    Read an image from file.
    
    Parameters
    ----------
    filename : String
        Path to a valid image file.
    flagColor : Boolean
        Value indicating to read the image with RGB (True) or Grayscale (False).

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
    im : Image (Numpy Array)
        Image to be printed to screen.
    title : String, optional
        Title of the image. The default is None.

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


def pintaIM(vim, title=None):
    """
    Print a list of images as one single picture.
    
    Parameters
    ----------
    vim : List of images
        A list containing images (Numpy Arrays), can be of any size and Grayscale or RGB.
    title : String, optional
        Title of the images. The default is None.

    Returns
    -------
    None.

    """
    
    white = (255, 255, 255)
    
    # Getting the maximum height of the list of images.
    maxHeight = max(i.shape[0] for i in vim)
    
    # Start to work on the fist image.
    if(len(vim[0].shape) == 2):
        vim[0] = np.uint8(vim[0])
        vim[0] = cv.cvtColor(vim[0], cv.COLOR_GRAY2BGR)    
    
    if(vim[0].shape[0] != maxHeight):
        strip = cv.copyMakeBorder(vim[0], 0, maxHeight-vim[0].shape[0], 0, 0, cv.BORDER_CONSTANT, value=white)       
    else:
        strip = vim[0]
    
    for i in vim[1:]:    
        
        if(len(i.shape) == 2):
            i = np.uint8(i)
            i = cv.cvtColor(i, cv.COLOR_GRAY2BGR)
        
        if(i.shape[0] != maxHeight):
            strip = cv.hconcat([strip, cv.copyMakeBorder(i, 0, maxHeight-i.shape[0], 0, 0, cv.BORDER_CONSTANT, value=white)])       
        else:
            strip = cv.hconcat([strip, i])

    pintaI(strip, title)


def pintaIMVentana(dictIm):
    size = len(dictIm)
    fig = plt.figure(figsize=(10,4))

    i = 1
    for element in dictIm:
        fig.add_subplot(1, size, i)
        im = dictIm[element]
        # Check if image is grayscale or RGB
        if len(im.shape) == 2:
            # Colormap it to grey and autonormalize values to between 0 and 1.
            plt.imshow(im, cmap='gray', norm=clr.Normalize())
        else:
            imAux = (im - np.min(im)) / (np.max(im) - np.min(im))
            plt.imshow(imAux[:,:,::-1])

        plt.title(element)
        i+=1
        
    fig.tight_layout()
    plt.show()

# MAIN
def main():
    """
    Main function from which all the other functions are to be called.

    Returns
    -------
    None.

    """

    #   Path the test images.
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
    
    #   Rest of images
    imageDave = leeImagen(srcDave, False)
    imageMessi = leeImagen(srcMessi, False)
    imageLogo = leeImagen(srcLogo, True)
   
    
    # Task 2: Visualize an arbitrary real number matrix
    ##########
    
    #   Generating random matrices with different ranges
    greyScaleMatrix = np.random.default_rng().uniform(-10.0, 220.0, (8,8))
    colorMatrix = np.random.default_rng().uniform(-10.0, 16.0, (8,8,3))
    
    """
    #   Showing the Orapples along with the matrices
    pintaI(imageOrappleGrey)
    pintaI(imageOrappleColor)
    pintaI(greyScaleMatrix)
    pintaI(colorMatrix)
    
    # Task 3: Concatenate multiple images into one
    ##########
    #   Constructing an image vector
    imageVector = []
    imageVector.append(imageDave)
    imageVector.append(imageMessi)
    imageVector.append(imageLogo)
    #   Showing the image vector as one
    pintaIM(imageVector)
    """
    
    # Task 4: Add 
    ##########
    
    # Task 5: Show multiple images in a single window with their own titles.
    ##########
    imageDict = {"Oof" : imageDave, "Pecho frío" : imageMessi, "Dolor" : imageLogo, "Sabroso" : imageOrappleColor, "Ruidoso" : colorMatrix}
    pintaIMVentana(imageDict)


# Setting up the script enviroment
if __name__ == "__main__":
    main()