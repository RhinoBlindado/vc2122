#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 1: Convolución y Derivadas
    Asignatura: Vision por Computador
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Octubre 2021
    
[ENGLISH]

    Practice 1: Convolution and Derivatives
    Course: Computer Vision
    Author: Valentino Lugli (Github: @RhinoBlindado)
    Octubre 2021

"""

# LIBRARIES

#   Using Matplotlib to show images
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#   Using OpenCV for everything else related to images.
import cv2 as cv

#   Using Numpy to manipulate images
import numpy as np

#   Using Math for more advanced calculations and functions
import math

# FUNCTIONS

### AUXILIAR FUNCTIONS ###

def Gray2Color(img):
    """
    Converts an Greyscale Image to BGR
    Parameters
    ----------
    img : Image (Numpy Array)
        The image from Grayscale.

    Returns
    -------
    Image (Numpy Array)
        The image converted to BGR.

    """
    return cv.cvtColor(np.uint8(img), cv.COLOR_GRAY2BGR)

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
    Image (Numpy Array)
        The image

    """
    return cv.imread(filename, int(flagColor))


def pintaI(im, title=None, norm=True):
    """
    Print an arbitrary real number matrix

    Parameters
    ----------
    im : Numpy Array
        Arbitrary number matrix to be normalized and printed.
    title : String, optional
        Title of the image. The default is None.
    norm : Boolean
        If True, normalize the image. Default is True.

    Returns
    -------
    None.

    """
    
    # Make each image appear on its own window with optional title.
    plt.figure()
    plt.title(title)
    # Check if image is grayscale or RGB
    if len(im.shape) == 2:
        # Colormap it to grey and autonormalize values to between 0 and 1.
        if norm:
            plt.imshow(im, cmap='gray', norm=clr.Normalize())
        else:
            plt.imshow(im, cmap='gray')
    else:
        # Normalize the color channels to between 0 and 1.
        if norm: imAux = (im - np.min(im)) / (np.max(im) - np.min(im))
        # Show the image with the channels flipped since OpenCV reads in BGR and Matplotlib shows in RGB.
        plt.imshow(imAux[:,:,::-1])
    plt.show()

def pintaIM(vim, title = None, color = (255, 255, 255)):
    """
    Print an horizontal list of images as one single picture.
    
    Parameters
    ----------
    vim : List of images
        A list containing images (Numpy Arrays), can be of any size and Grayscale or RGB.
    title : String, optional
        Title of the whole image. The default is None.
    color : BGR tuple.
        Color for the padding.

    Returns
    -------
    None.

    """    
    # Getting the maximum height of the list of images.
    maxHeight = max(i.shape[0] for i in vim)
    
    # This implementantions takes the biggest image and makes that the maximum height of the picture strip
    # therefore any image that is smaller than this height will be added in their original size, 
    # the rest of the space will be padded with a color, in this case, white.
    
    # Start to work on the first image, if it's grayscale convert it to BGR.
    if(len(vim[0].shape) == 2): vim[0] = Gray2Color(vim[0])   
    
    # If the image doesn't have the max height, add white padding vertically.
    if(vim[0].shape[0] != maxHeight):
        strip = cv.copyMakeBorder(vim[0], 0, maxHeight-vim[0].shape[0], 0, 0, cv.BORDER_CONSTANT, value=color)       
    else:
        strip = vim[0]
    
    # Repeat this process for the rest of the images vector.
    for i in vim[1:]:    

        # If grayscale, convert it to BGR.        
        if(len(i.shape) == 2): i = Gray2Color(i)
        
        # In adition to adding padding if needed, now concatenate horizontally the pictures.
        if(i.shape[0] != maxHeight):
            strip = cv.hconcat([strip, cv.copyMakeBorder(i, 0, maxHeight-i.shape[0], 0, 0, cv.BORDER_CONSTANT, value=color)])       
        else:
            strip = cv.hconcat([strip, i])

    # Once it's done, print the image strip as one picture.

    plt.title(title)
    plt.imshow(strip[:,:,::-1])


def cambiarColor(imagen, listaPuntos, color):
    """
    Changes the color of an image, given a list of points and a given color.
    
    Parameters
    ----------
    imagen : Image (Numpy Array)
        The base image to be modified.
    listaPuntos : Vector of 2D Numpy Arrays
        The list of points that are to be changed in color.
    color : 3D tuple
        The color to paint the image in the list of points.

    Returns
    -------
    imagen : Image (Numpy Array)
        Returns the modified image.

    """
    if(len(imagen.shape) == 2): imagen = Gray2Color(imagen)
    
    for coord in listaPuntos:
        if(0 <= coord[0] < imagen.shape[0] and 0 <= coord[1] < imagen.shape[1]):
            imagen[coord[0], coord[1], :] = color
        
    return imagen


def pintaIMVentana(dictIm, title=None):
    """
    Prints a series of images in the same window but each image has its own title.

    Parameters
    ----------
    dictIm : Dictionary of "Title : Image"
        Dictionary containg a Title that maps to an Image object (Numpy Array) to be printed.

    Returns
    -------
    None.

    """
    # Get the size of the dictionary
    size = len(dictIm)
    # Setting the size of the window.
    fig = plt.figure(figsize=(10,4))
    
    # For each element in the dictionary...
    i = 1
    for element in dictIm:
        # ... add a subplot horizontally for each image; index 'i' is an identifier..
        fig.add_subplot(1, size, i)
        im = dictIm[element]
        # ...Add the image to the subplot, same prodecure as normal printing, ...
        if len(im.shape) == 2:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im[:,:,::-1])
        # ... Add the title...
        plt.title(element)
        # ... And update the identifier (just a counter, really).
        i+=1
    
    # Set up the layout.
    fig.tight_layout()
    # Show the image to screen.
    fig.suptitle(title, weight='bold')
    plt.show()


### MAIN CODE ###

#   Path the test images.
imgCat = np.float64(leeImagen("./data/cat.bmp", False))
imgMotorBike = np.float64(leeImagen("./data/motorcycle.bmp", False))
imgChicote = np.float64(leeImagen("./extraPics/chicote.jpeg", False))
imgBike = np.float64(leeImagen("./data/bicycle.bmp", False))
imgCircle = np.float64(leeImagen("./extraPics/circle.png", False))
imgPixels = np.float64(leeImagen("./extraPics/pixels.jpg",False))

## Exercise 1A: Generate gaussian blur and derivative masks.

def gaussian(x, sigma):
    return math.exp(-((pow(x, 2))/(2*pow(sigma, 2))))

def gaussianFirstD(x, sigma):
    return -(x*gaussian(x, sigma))/(pow(sigma, 2))

def gaussianSecondD(x, sigma):
    return (gaussian(x, sigma) * (pow(x, 2) - pow(sigma, 2)))/(pow(sigma, 4))

def gaussianMask(dx, sigma = None, maskSize = None):
    
    mask = None
    if((0<= dx and dx < 3) and (sigma != None or maskSize != None)):
        # If sigma is passed, calculate the mask size, even if the mask is passed.
        # This is so that if a mask doesn't match the sigma, sigma takes priority.
        if(sigma != None):
            maskSize = 2 * (3 * math.ceil(sigma)) + 1
        # If sigma isn't passed, but the mask is, then calculate the sigma for the given mask.
        else:
            sigma = (maskSize - 1) / 6 
    
        k = (maskSize - 1) / 2
    
        if(dx == 0):
            mask = [gaussian(x,sigma) for x in np.arange(-k, k + 1)]
            mask /= np.sum(mask)
        elif(dx == 1):
            mask = [gaussianFirstD(x, sigma) for x in np.arange(-k, k + 1)]
        elif(dx == 2):
            mask = [gaussianSecondD(x, sigma) for x in np.arange(-k, k + 1)]
    
        mask = np.fromiter(mask, float, len(mask))
    
    return mask

def plotGauss(maskSize):
    
    k = (maskSize - 1) / 2
    y = np.arange(-k, k + 1)
    
    # Getting the masks from own functions.
    impl = [ gaussianMask(0, None, maskSize), gaussianMask(1, None, maskSize), gaussianMask(2, None, maskSize) ]
    
    # Getting OpenCV kernels for the same mask size and derivatives.
    opcv = [ cv.getDerivKernels(0, 1, maskSize)[0], cv.getDerivKernels(1, 1, maskSize)[0],  cv.getDerivKernels(2, 1, maskSize)[0] ]
    titles = ['Kernel Gaussiano L='+ str(maskSize), 'Kernel Gaussiano L='+ str(maskSize)+', 1ª Derivada', 'Kernel Gaussiano L='+ str(maskSize)+', 2ª Derivada']
    
    # Printing the result...
    for i in range(3):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1.plot(y, impl[i], '.-y')
        ax1.set_title("Implementación")
    
        ax2.plot(y, opcv[i], '.-r')
        ax2.set_title("OpenCV")
        fig.suptitle(titles[i], weight='bold', ha='left')
        plt.show()


# print("Ejercicio 1A: Generando máscaras de alisamiento y de derivadas")

# print(" -Imprimiendo comparativa entre implementación y OpenCV...",end='')
# Generating 3 masks with length 5, 7 and 9.
gaussMask5 = gaussianMask(0, None, 5)
gaussMask7 = gaussianMask(0, None, 7)
gaussMask9 = gaussianMask(0, None, 9)

# Generating the same three masks with OpenCV.

# Plotting their differences:
    
# plotGauss(5)
# plotGauss(7)
# plotGauss(9)
# print("Listo")


#   Exercise 1B

def convolve(f, g):
    midPoint = math.floor((len(g)-1)/2)
    convSize = len(f) + len(g) - 1
    h = np.zeros(convSize)
    f = np.pad(f, int((convSize - len(f))/2))
    for x in range(0, len(f)):
        val = 0
        for i in range(0, len(g)):
            if(0 <= x+midPoint-i and x+midPoint-i < len(f)):
                val += g[i] * f[x + midPoint - i]
        h[x] = val

    return h

# print("\nEjercicio 1B")
# binomial = np.array([1, 2, 1])
# basicDeriv = np.array([-1, 0, 1])

# size5Kernel = convolve(binomial, binomial)
# size5Deriv = convolve(basicDeriv, binomial)

# print(" -Dada la aproximación binobial a la Gaussiana",binomial, "se puede obtener una máscara de",
#       "longitud 5 realizando la convolución de la binomial consigo misma:",size5Kernel)
# print(" -De la misma manera, se puede convolucionar", size5Kernel, "con la binomial para",
#       "obtener la máscara de longitud 7: ", convolve(size5Kernel, binomial))
# print("-Si se convoluciona la binomial con la máscara derivada",basicDeriv,"se obtiene la máscara",
#       "de derivación de longitud 5:", size5Deriv, " y este resultado con la ")



#   Exercise 1C

def addPadding(img, sizePadding, typePadding, color=None):

    if(cv.BORDER_CONSTANT):
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding, value=color)
    else:
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding)

    return paddedImg


def convolveImage(img, hMask, vMask):
    
    hMid = math.floor((len(hMask)-1)/2)
    vMid = math.floor((len(vMask)-1)/2)
    
    tempImg = np.zeros([img.shape[0], img.shape[1] - (hMid * 2)])
    convImg = np.zeros([img.shape[0] - (vMid * 2), img.shape[1] - (vMid * 2)])
       
    
    for x in range(hMid, img.shape[0] - hMid):
       for y in range(hMid, img.shape[1] - hMid):
            val = 0
            for i in range(len(hMask)):
                val += hMask[i] * img[x][y+hMid-i]
            tempImg[x][y-hMid] = val
                
    for x in range(vMid, tempImg.shape[0] - vMid):
        for y in range(tempImg.shape[1]):
            val = 0
            for i in range(len(vMask)):
                val += vMask[i] * tempImg[x + vMid - i][y]
            convImg[x-vMid][y] = val
    
    return convImg 

def getDiff(imgA, imgB):
    return np.mean(np.abs(imgA - imgB))

# # Set a picture, sigma and the mask.
# oImg = imgMotorBike
# sigma = 2
# mask = gaussianMask(0, sigma)
# deriv = gaussianMask(1, sigma)

# # Add padding to the image original image.
# padSize = math.floor((len(mask)-1)/2)
# imgPad = addPadding(oImg, padSize, cv.BORDER_REFLECT)

# # Blur using own function.
# imgConv = convolveImage(imgPad, mask, mask)

# # Blur with OpenCV
# imgCVConv = cv.GaussianBlur(oImg, (0,0), sigma)

# # Show the images:
# pintaI(oImg, "Imagen original")
# pintaIMVentana({"Implementación" : imgConv, "OpenCV" : imgCVConv}, "Implementación vs OpenCV")

# print("La diferencia media es",getDiff(imgConv, imgCVConv))

# imgdx = convolveImage(oImg, deriv, mask)
# imgdy = convolveImage(oImg, mask, deriv)
# pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy}, "Derivada por Ejes")

# Exercise 1D

def laplacian(img, sigma):
    gauss = gaussianMask(0, sigma) 
    gdxx = gaussianMask(2, sigma)
    dxx = convolveImage(img, gauss, gdxx)
    dyy = convolveImage(img, gdxx, gauss)
    
    L = pow(sigma, 2) * (dxx + dyy)
    return L
    

# lapImage = imgBike
# sigma = 0.75

# lapImage = addPadding(lapImage, math.ceil(sigma) * 3, cv.BORDER_REFLECT)
# a = cv.Laplacian(imgBike, cv.CV_64F)
# pintaI(a)
# b = laplacian(lapImage, sigma)
# pintaI(b)


# Exercise 2A

def subSample(img):
    result = img[::2, ::2]
    return result

def gaussianPyramid(img, maxLevel, sigma=1):
    
    mask = gaussianMask(0, 1)
    imgList = [img]
    
    if (maxLevel > 0):
        for i in range(0, maxLevel):
            tempI = convolveImage(img, mask, mask)
            tempI = addPadding(tempI, 3, cv.BORDER_REFLECT)
            img = subSample(tempI)
            imgList.append(img)
            
    return imgList
            
baseImg = imgPixels
gaussPyrImg = addPadding(baseImg, 3, cv.BORDER_REFLECT)
gaussPyrLst = gaussianPyramid(gaussPyrImg, 4)
pintaIM(gaussPyrLst)
pintaI(gaussPyrLst[3])

