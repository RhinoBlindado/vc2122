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

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


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
    img = normalize(img)
    img = cv.cvtColor(np.uint8(img * 255), cv.COLOR_GRAY2BGR)
    return img

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
    return np.float64(cv.imread(filename, int(flagColor)))


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
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

def pintaIM(vimIn, title = None, color = (255, 255, 255)):
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
    
    vim = vimIn.copy()
    # Getting the maximum height of the list of images.
    maxHeight = max(i.shape[0] for i in vim)
    
    # This implementantions takes the biggest image and makes that the maximum height of the picture strip
    # therefore any image that is smaller than this height will be added in their original size, 
    # the rest of the space will be padded with a color, in this case, white.
    
    # Start to work on the first image, if it's grayscale convert it to BGR.
    if(len(vim[0].shape) == 2): 
        vim[0] = Gray2Color(vim[0])
    else:
        vim[0] = normalize(vim[0])
    
    # If the image doesn't have the max height, add white padding vertically.
    if(vim[0].shape[0] != maxHeight):
        strip = cv.copyMakeBorder(vim[0], 0, maxHeight-vim[0].shape[0], 0, 0, cv.BORDER_CONSTANT, value=color)       
    else:
        strip = vim[0]
    
    # Repeat this process for the rest of the images vector.
    for i in vim[1:]:    

        # If grayscale, convert it to BGR.        
        if(len(i.shape) == 2): 
            i = Gray2Color(i)
        else:
            i = normalize(i)
        
        # In adition to adding padding if needed, now concatenate horizontally the pictures.
        if(i.shape[0] != maxHeight):
            strip = cv.hconcat([strip, cv.copyMakeBorder(i, 0, maxHeight-i.shape[0], 0, 0, cv.BORDER_CONSTANT, value=color)])       
        else:
            strip = cv.hconcat([strip, i])

    # Once it's done, print the image strip as one picture.
    plt.figure()
    plt.xticks([])
    plt.yticks([])
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
        plt.xticks([])
        plt.yticks([])
        im = dictIm[element]
        # ...Add the image to the subplot, same prodecure as normal printing, ...
        if len(im.shape) == 2:
            plt.imshow(im, cmap='gray', norm=clr.Normalize())
        else:
            im = normalize(im)
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


### MAIN ###

#   Ruta para las imágenes.
imgCat = leeImagen("./imagenes/cat.bmp", False)
imgDog = leeImagen("./imagenes/dog.bmp", False)

imgEinstein = leeImagen("./imagenes/einstein.bmp", False)
imgMonroe = leeImagen("./imagenes/marilyn.bmp", False)

imgBird = leeImagen("./imagenes/bird.bmp", False)
imgPlane = leeImagen("./imagenes/plane.bmp", False)

imgMotorBike = leeImagen("./imagenes/motorcycle.bmp", False)
imgBike = leeImagen("./imagenes/bicycle.bmp", False)

#%%
## Ejercicio 1A

def gaussian(x, sigma):
    """
    Calcular la función gaussiana

    Parameters
    ----------
    x : Numero flotante
        Punto en el eje X donde se desea calcular la gaussiana
    sigma : Numero flotante
        Valor del sigma de la función

    Returns
    -------
    Flotante
        Cálculo de la función en X con el sigma dado.

    """
    return math.exp(-((pow(x, 2))/(2*pow(sigma, 2))))

def gaussianFirstD(x, sigma):
    return -(x*gaussian(x, sigma))/(pow(sigma, 2))

def gaussianSecondD(x, sigma):
    return (gaussian(x, sigma) * (pow(x, 2) - pow(sigma, 2)))/(pow(sigma, 4))

def gaussianMask(dx, sigma = None, maskSize = None):
    """
    Obtener una máscara gaussiana, su primera o segunda derivada dado su sigma o el tamaño.
    Parameters
    ----------
    dx : Número de 0 a 2.
        La derivada deseada, si es 0 se obtiene la función gaussiana.
    sigma : Flotante, opcional
        El valor de sigma deseado.
    maskSize : Entero, opcional.
        El tamaño deseado de la máscara, si se pasa junto al sigma el valor es desestimado.

    Returns
    -------
    mask : Vector 1D
        Vector conteniendo una máscara gaussiana discreta.
    """
    
    mask = None
    # Si no se cumplen las condiciones, se devuelve una máscara vacía.
    if((0<= dx and dx < 3) and (sigma != None or maskSize != None)):
        # Si se pasa el sigma, calcular el tamaño de la máscara incluso si se pasa el tamaño.
        # Esto es así en caso de que el tamaño no sea el adecuado para un sigma dado.
        if(sigma != None):
            maskSize = 2 * (3 * math.ceil(sigma)) + 1
        # Si no se pasa el sigma, se calcula con el tamaño de la máscara.
        else:
            sigma = (maskSize - 1) / 6 
    
        # Se obtiene el tamaño del lado de la máscara para hacer la máscara.
        k = (maskSize - 1) / 2
    
        if(dx == 0):
            # Si es la máscara gaussiana, luego de realizar el bucle de -k hasta k, 
            # se normaliza la máscara para que sume 1.
            mask = [gaussian(x,sigma) for x in np.arange(-k, k + 1)]
            mask /= np.sum(mask)
        elif(dx == 1):
            mask = [gaussianFirstD(x, sigma) for x in np.arange(-k, k + 1)]
        elif(dx == 2):
            mask = [gaussianSecondD(x, sigma) for x in np.arange(-k, k + 1)]
    
        # Se convierte el vector de Python a un vector de Numpy.
        mask = np.fromiter(mask, float, len(mask))
    
    return mask

def plotGauss(maskSize):
    
    k = (maskSize - 1) / 2
    y = np.arange(-k, k + 1)
    
    # Obteniendo máscaras de la función propia.
    impl = [ gaussianMask(0, None, maskSize), gaussianMask(1, None, maskSize), gaussianMask(2, None, maskSize) ]
    
    # Obteniendo las máscaras de OpenCV para el mismo tamaño y derivada.
    opcv = [ cv.getDerivKernels(0, 1, maskSize)[0], cv.getDerivKernels(1, 1, maskSize)[0],  cv.getDerivKernels(2, 1, maskSize)[0] ]
    titles = ['Kernel L='+ str(maskSize), 'Kernel L='+ str(maskSize)+', 1ª Derivada', 'Kernel L='+ str(maskSize)+', 2ª Derivada']
    
    # Imprimiendo el resultado:
    for i in range(3):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1.plot(y, impl[i], '.-y')
        ax1.set_title("Implementación")
    
        ax2.plot(y, opcv[i], '.-r')
        ax2.set_title("OpenCV")
        fig.suptitle(titles[i], weight='bold')
        plt.show()


print("---Ejercicio 1A---")

print(" -Imprimiendo comparativa entre implementación y OpenCV...",end='')

# Se generan las máscaras gaussianas implementadas y las de OpenCV en la función.    
plotGauss(5)
plotGauss(7)
plotGauss(9)
print("Listo")

#%%
#   Ejericio 1B

print("\n---Ejercicio 1B---")
binomial = np.array([1, 2, 1])
basicDeriv = np.array([-1, 0, 1])

size5Kernel = np.convolve(binomial, binomial)
size5Deriv = np.convolve(basicDeriv, binomial)
size7Kernel = np.convolve(size5Kernel, binomial)
size7Deriv = np.convolve(size5Deriv, binomial)
print("- Dada la aproximación binobial a la Gaussiana",binomial, "se puede obtener una máscara de",
      "longitud 5 realizando la convolución de la binomial consigo misma:",size5Kernel)
print("- De la misma manera, se puede convolucionar", size5Kernel, "con la binomial para",
      "obtener la máscara de longitud 7: ", size7Kernel)
print("- Si se convoluciona la binomial con la máscara derivada",basicDeriv,"se obtiene la máscara",
      "de derivación de longitud 5:", size5Deriv, " y este resultado con la binomial se obtiene:", size7Deriv)

cvKernels = cv.getDerivKernels(0, 1, 9)

print("- Los kernels obtenidos por OpenCV para una máscara tamaño 9, sin derivar en x y primera derivada en y son:",np.transpose(cvKernels[0]), "y", np.transpose(cvKernels[1]),".")
print("- Esto se puede obtener realizando la convolución nuevamente de las máscaras de tamaño 7 (de alisamiento y derivada) con la binomial:",np.convolve(size7Kernel, binomial),"y la derivada:", np.convolve(size7Deriv, binomial))

#%%
#   Ejercicio 1C
print("\n---Ejercicio 1C---")


def addPadding(img, pSigma = None, pMask = None, typePadding = cv.BORDER_REFLECT, color=None):
    """
    Añadir padding a una imagen, dado su sigma o máscara.
    Se asume que será el mismo valor de sigma horizontal y verticalmente.

    Parameters
    ----------
    img : Imagen
    pSigma : Flotante, optional
        Sigma que uilizan las máscaras.
    pMask : Entero, optional
        Longitud de la máscara. Si se pasa junto a pSigma, el valor es descartado.
    typePadding : Tipos de Padding de OpenCV, optional
        Definir el tipo de padding. Por defecto es cv.BORDER_REFLECT.
    color : 3-tupla de 0 a 255, optional
        Color del borde si se utiliza cv.BORDER_CONSTANT. Por defecto está a None.

    Returns
    -------
    paddedImg : Imagen
        Imagen con padding añadido.

    """
    
    # Calculando el tamaño del padding, lo que se obtiene calculando que tan grande
    # sería el lado de una máscara, ya sea por sigma o por el tamaño general.
    if(pSigma != None):
        sizePadding = math.ceil(pSigma) * 3
    else:
        sizePadding = math.floor((pMask-1)/2)
    

    if(cv.BORDER_CONSTANT):
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding, value=color)
    else:
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding)

    return paddedImg


def convolveImage(img, xMask, yMask):
    """
    Convolucionar una imagen con dos máscaras.

    Parameters
    ----------
    img : Imagen
        Imagen a convolucionar
    xMask : Numpy Array
        Máscara para convolucionar horizontalmente.
    yMask : Numpy Array
        Máscara para convolucionar verticalmente.

    Returns
    -------
    convImg : Imagen
        Imagen convolucionada

    """
    # Nota: Si bien no se especificó, esta función soporta máscaras de distintos tamaños.
    
    # Obtener la longitud de las máscaras
    lenH = len(yMask)
    lenV = len(xMask)

    # Obtener el punto central de las máscaras.
    hMid = math.floor((lenH-1)/2)
    vMid = math.floor((lenV-1)/2)
        
    # Obtener las dimensiones de la imagen sin el padding, ya que la 
    # convolución reducirá el tamaño de la imagen a como estaba sin padding.
    x = img.shape[0] - hMid * 2
    y = img.shape[1] - hMid * 2
    
    # Para acelerar el cálculo, se repiten las máscaras por la longitud
    # de la imagen, se utiliza reshape para poner verticales las máscaras
    # y luego se repiten tantas veces por columnas los valores que poseen.
    hMask = np.repeat(np.reshape(yMask, (len(yMask), 1)), y, 1)
    vMask = np.repeat(np.reshape(xMask, (len(xMask), 1)), x, 1)

    # Se crea una imagen temporal intermedia copiando la imagen original.
    tempImg = np.array(img)
    
    # Los cálculos varian ligeramente si la imagen está en escala de grises o 
    # a color, se determina y se hacen los cálculos acordes.
    if(len(img.shape) == 2):
    
        # Se crea la imagen final, una matriz vacía del tamaño original.
        convImg = np.empty((x,y))
        
        # Se realiza la primera convolución de manera vertical:
        #  - Por cada fila de la imagen, se multiplica la máscara repetida por una "submatriz" de la imagen de las mismas dimensiones.
        #  - Esta multiplicación se suma por columnas y da como resultado la convolución de esa fila.
        #  - La imagen resultante se almacena en tempImg, se realiza un desfase en la imagen para mantener el padding de la imagen original pero reemplazando los píxeles "canonicos".
        for i in range(0, x):
            tots = np.sum(img[i: i + lenH, hMid:y + hMid] * hMask, 0)
            tempImg[i + hMid, hMid: y + hMid] = tots
    
        # Se transpone la imagen para repetir el mismo proceso en horizontal.
        tempImg = np.transpose(tempImg)
        
        # Segunda convolución en "horizontal".
        # Mismo procedimiento que el explicado anteriormente.
        for i in range(0, y):
            tots = np.sum(tempImg[i: i + lenV, vMid:x + vMid] * vMask, 0)
            convImg[:,i] = tots
    else:
        # Esto fue implementado para el Bonus
        
        # Variante para colores, se repite la máscara en las tres bandas de colores.
        hMask = np.repeat(hMask[:, :, np.newaxis], 3, axis=2)
        vMask = np.repeat(vMask[:, :, np.newaxis], 3, axis=2)
        
        # Proceso similar al anterior, ahora con las 3 bandas incluidas.
        convImg = np.empty((x,y,3))
        
        # Ídem a su versión en escala de grises, ahora se incluye la tercera 
        # columna de los colores.
        for i in range(0, x):
            tots = np.sum(img[i: i + lenH, hMid:y + hMid, :] * hMask, 0)
            tempImg[i + hMid, hMid: y + hMid, :] = tots
    
        # Se utiliza swapaxes para realizar la transposición del eje X e Y.
        tempImg = np.swapaxes(tempImg, 0, 1)
        
        # Ídem a lo anterior.
        for i in range(0, y):
            tots = np.sum(tempImg[i: i + lenV, vMid:x + vMid, :] * vMask, 0)
            convImg[:,i,:] = tots
    
    return convImg
        


def getDiff(imgA, imgB):
    """
    Obtener el error cuadrado de dos imágenes.

    Parameters
    ----------
    imgA : TYPE
        DESCRIPTION.
    imgB : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.mean(np.sqrt(pow(imgA - imgB, 2)))


# Se elige una imagen, un sigma y las máscaras.
oImg = imgMotorBike
sigma = 2
mask = gaussianMask(0, sigma)
deriv = gaussianMask(1, sigma)

# Se añade el padding.
imgPad = addPadding(oImg, None, len(mask), cv.BORDER_REFLECT)

# Se desenfoca con la función implementada.
imgConv = convolveImage(imgPad, mask, mask)

# Se compara con el desenfoque que trae OpenCV.
imgCVConv = cv.GaussianBlur(oImg, (0,0), sigma, borderType=cv.BORDER_REFLECT)

# Mostrando las imágenes:
pintaI(oImg, "Imagen original")
pintaI(imgConv, "Implementación")
pintaI(imgCVConv, "OpenCV")

print("La diferencia media es",getDiff(imgConv, imgCVConv))

# Generando 3 máscaras con longitud 5, 7 y 9 con la función del ejercicio A.
gaussMask5 = gaussianMask(0, None, 5)
gaussMask7 = gaussianMask(0, None, 7)
gaussMask9 = gaussianMask(0, None, 9)

gaussDeri5 = gaussianMask(1, None, 5)
gaussDeri7 = gaussianMask(1, None, 7)
gaussDeri9 = gaussianMask(1, None, 9)

# Generando imágenes de derivadas con esas máscaras:
imgdx = convolveImage(imgPad, gaussDeri5, gaussMask5)
imgdy = convolveImage(imgPad, gaussMask5, gaussDeri5)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy, "Manginutd de Gradiente" : mixed}, "Derivada por Ejes, T=5")
pintaI(mixed, "Magnitud de Gradiente")

imgdx = convolveImage(imgPad, gaussDeri7, gaussMask7)
imgdy = convolveImage(imgPad, gaussMask7, gaussDeri7)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy, "Manginutd de Gradiente" : mixed}, "Derivada por Ejes, T=7")
pintaI(mixed, "Magnitud de Gradiente")


imgdy = convolveImage(imgPad, gaussMask9, gaussDeri9)
imgdx = convolveImage(imgPad, gaussDeri9, gaussMask9)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy, "Manginutd de Gradiente" : mixed}, "Derivada por Ejes, T=9")


#%%
# Exercise 1D
def laplacianMask(sigma = None, maskSize = None):
    gauss = gaussianMask(0, sigma, maskSize)
    gdxx = gaussianMask(2, sigma, maskSize)
    
    dxx = np.outer(gauss, gdxx)
    dyy = np.outer(gdxx, gauss)
    
    if(sigma == None):
        sigma = (maskSize - 1) / 6 
    
    L = pow(sigma, 2) * (dxx + dyy)
    
    return L

def laplacian(img, sigma):
    gauss = gaussianMask(0, sigma)
    gdxx = gaussianMask(2, sigma)
    dxx = convolveImage(img, gdxx, gauss)
    dyy = convolveImage(img, gauss, gdxx)
    
    dxx = normalize(dxx)
    dyy = normalize(dyy)
    
    L = pow(sigma, 2) * (dxx + dyy)
    return L
    

lapImage = imgCat
lapSigma1 = 1
lapSigma3 = 3

lapConv1 = addPadding(lapImage, lapSigma1, None, cv.BORDER_REFLECT)
lapResult1 = laplacian(lapConv1, lapSigma1)
lapResultCV1 = cv.Laplacian(lapImage, cv.CV_64F, ksize = 7, borderType=cv.BORDER_REFLECT)


pintaI(lapResult1, "Laplaciano: Implementación (σ=" + str(lapSigma1)+")")
pintaI(lapResultCV1, "Laplaciano: OpenCV")

lapConv3 = addPadding(lapImage, lapSigma3, None, cv.BORDER_REFLECT)
lapResult3 = laplacian(lapConv3, lapSigma3)
lapResultCV3 = cv.Laplacian(lapImage, cv.CV_64F, ksize = 19)

pintaI(lapResult3, "Laplaciano: Implementación (σ=" + str(lapSigma3)+")")
pintaI(lapResultCV3, "Laplaciano: OpenCV")


lapMask1 = laplacianMask(maskSize=3)
lapMask3 = laplacianMask(3)

print("Máscara Laplaciana, σ=1\n", lapMask1)



#%%
# Exercise 2A

def subSample(img):
    result = img[::2, ::2]
    return result

def gaussianPyramid(img, maxLevel, sigma=1):
        
    if (maxLevel > 0):
        mask = gaussianMask(0, sigma)
        imgList = [img]
        
        for i in range(0, maxLevel):
            tempI = addPadding(img, None, len(mask), cv.BORDER_REFLECT)
            tempI = convolveImage(tempI, mask, mask)
            img = subSample(tempI)
            imgList.append(img)
    else:
         imgList = None       
         
    return imgList

def gaussianPyrCV(img, maxLevel):
    
    imgList = [img]
    for i in range(0, maxLevel):
        tempI = cv.pyrDown(img, borderType=cv.BORDER_REFLECT)
        img = tempI
        imgList.append(img)
        
    return imgList

baseImg = imgBike
gaussPyrLst = gaussianPyramid(baseImg, 4, 1)

pintaIM(gaussPyrLst, title="Implentación")
# pintaI(gaussPyrLst[-1])

gaussPyrCV = gaussianPyrCV(baseImg, 4)
pintaIM(gaussPyrCV,title="OpenCV")
# pintaI(gaussPyrCV[-1])

#%%
# Exercise 2B
def laplacianPyramid(gaussPyr):
    
    gaussPyr = np.flip(gaussPyr)
    maxLevel = len(gaussPyr)
    imgList  = [gaussPyr[0]]
    
    for i in range(1, maxLevel):
        expandedImg = cv.resize(gaussPyr[i-1], (gaussPyr[i].shape[1], gaussPyr[i].shape[0]), interpolation=cv.INTER_LINEAR)
        tempI = gaussPyr[i] - expandedImg
        imgList.append(tempI)
    
    return imgList
    
def laplacianPyrCV(gaussPyr):
    gaussPyr = np.flip(gaussPyr)
    maxLevel = len(gaussPyr)
    imgList  = [gaussPyr[0]]
    
    for i in range(1, maxLevel):
        expandedImg = cv.pyrUp(gaussPyr[i-1], dstsize=(gaussPyr[i].shape[1], gaussPyr[i].shape[0]))
        tempI = gaussPyr[i] - expandedImg
        imgList.append(tempI)
    
    return imgList

lapPyrLst = laplacianPyramid(gaussPyrLst)
lapPyrCV = laplacianPyrCV(gaussPyrCV)

pintaIM(lapPyrLst, "Pirámide Laplaciana, Implementación")
pintaIM(lapPyrCV, "Pirámide Laplaciana, OpenCV")

#%%
# Exercise 2C
print("\n---Ejercicio 2C---")

def recoverImg(lapPyr):
   maxLevel = len(lapPyr)
   baseImg = lapPyr[0]
   
   for i in range(1, maxLevel):
        expandedImg = cv.resize(baseImg, (lapPyr[i].shape[1], lapPyr[i].shape[0]), interpolation=cv.INTER_LINEAR)
        baseImg = lapPyr[i] + expandedImg
        
   return baseImg
   
recoveredImg = recoverImg(lapPyrLst)

pintaIMVentana({"Imagen Original" : baseImg, "Imagen Recuperada" : recoveredImg}, "Ejercicio 2C")
print("La diferencia de la imagen original y la recuperada es", getDiff(baseImg, recoveredImg))
    
#%%
# Bonus 1

def genHybridImg(highF, hFSigma, lowF, lFSigma):
        
    lowMask = gaussianMask(0, lFSigma, None)    

    highF = addPadding(highF, hFSigma, None, cv.BORDER_REFLECT)
    lowF = addPadding(lowF, lFSigma, None, cv.BORDER_REFLECT)
    
    highPass = laplacian(highF, hFSigma) * -1
    lowPass = convolveImage(lowF, lowMask, lowMask)

    highPass = normalize(highPass)
    lowPass = normalize(lowPass)
    
    hybrid = highPass + lowPass
        
    imgDict = {"Alta Frecuencia" : highPass, "Baja Frecuencia" : lowPass, "Híbrida" : hybrid}
    return hybrid, imgDict
    

hyb, imgD = genHybridImg(imgBike, 1.5, imgMotorBike, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)

hyb, imgD = genHybridImg(imgBird, 1.5, imgPlane, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)

hyb, imgD = genHybridImg(imgCat, 2, imgDog, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)


#%%
# Bonus 2

imgCatRGB = leeImagen("./imagenes/cat.bmp", True)
imgDogRGB = leeImagen("./imagenes/dog.bmp", True)

imgBirdRGB = leeImagen("./imagenes/bird.bmp", True)
imgPlaneRGB = leeImagen("./imagenes/plane.bmp", True)

imgMotorBikeRGB = leeImagen("./imagenes/motorcycle.bmp", True)
imgBikeRGB = leeImagen("./imagenes/bicycle.bmp", True)


hyb, imgD = genHybridImg(imgBikeRGB, 1.25, imgMotorBikeRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)


hyb, imgD = genHybridImg(imgBirdRGB, 1.5, imgPlaneRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)


hyb, imgD = genHybridImg(imgCatRGB, 2, imgDogRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr)


#%%
# Bonus 3

imgDaVinci = leeImagen("./imagenes/davinci.jpg", True)
imgGioconda = leeImagen("./imagenes/gioconda.jpg", True)

hyb, imgD = genHybridImg(imgDaVinci, 1, imgGioconda, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIM(imgPyr)

imgModelT = leeImagen("./imagenes/modelt.bmp", True)
imgTeslaX = leeImagen("./imagenes/tesla.bmp", True)

hyb, imgD = genHybridImg(imgModelT, 1, imgTeslaX, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIM(imgPyr)