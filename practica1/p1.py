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

# LIBRERÍAS

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

### FUNCIONES AUXILIARES ###

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


### COMIENZO EJERCICIOS ###
# Para ejecutar cada celda, hacer Ctrl+Enter en la celda resaltada o utilizando
# el icono justo a la derecha del icono verde de ejecución completa del código.

# Ejecutar esta celda primero para tener las funciones auxiliares en memoria.
# Las celdas primero deben ejecutarse secuencialmente, luego, se pueden ejecutar en cualquier orden.

# También se puede ejecutar todo el código de una, pues las funciones de dibujado no sobreescriben imágenes.

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

def convolveImage(oImg, xMask, yMask, borderType = cv.BORDER_REFLECT, color = None):
    """
    Convolucionar una imagen con dos máscaras.

    Parameters
    ----------
    oImg : Imagen
        Imagen a convolucionar
    xMask : Numpy Array
        Máscara para convolucionar horizontalmente.
    yMask : Numpy Array
        Máscara para convolucionar verticalmente.
    borderType : OpenCV Border
        Tipo de borde, por defecto es reflejar bordes.

    Returns
    -------
    convImg : Imagen
        Imagen convolucionada

    """
    # Notas: 
    #       - Si bien no se especificó, esta función soporta máscaras de distintos tamaños.
    #       - Como la convolución y el padding están muy relacionados, se pensó que es mejor incluir directamente el padding en la función.
    
    # Obtener las dimensiones de la imagen 
    x = oImg.shape[0]
    y = oImg.shape[1]
    
    # Obtener la longitud de las máscaras
    lenH = len(yMask)
    lenV = len(xMask)

    # Obtener el punto central de las máscaras, que también es tamaño de su lado.
    hMid = math.floor((lenH-1)/2)
    vMid = math.floor((lenV-1)/2)
    
    # Añadir el padding para el lado que va a ser convolucionado.
    img = cv.copyMakeBorder(oImg, hMid, hMid, 0, 0, borderType, value=color)
    
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
        #  - Por cada fila de la imagen, se multiplica la máscara repetida por una "submatriz" de la imagen de las mismas dimensiones que dicha máscara.
        #  - Esta multiplicación se suma por columnas y da como resultado la convolución de una fila.
        #  - La imagen resultante se almacena en tempImg, se realiza un desfase en la imagen para mantener el padding de la imagen original pero reemplazando los píxeles "canonicos".
        for i in range(0, x):
            tots = np.sum(img[i: i + lenH, :] * hMask, 0)
            tempImg[i + hMid, :] = tots
    
        # Se transpone la imagen para repetir el mismo proceso en horizontal.
        tempImg = np.transpose(tempImg)
        # Se añade el padding para el otro lado, a menos que se utilize BORDER_CONSTANT, el padding estará convolucionado por ser una extensión de la imagen que ya se convolucionó por un lado.
        tempImg = cv.copyMakeBorder(tempImg, vMid, vMid, 0, 0, borderType, value=color)
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
            tots = np.sum(img[i: i + lenH, :, :] * hMask, 0)
            tempImg[i + hMid, :, :] = tots
    
        # Se utiliza swapaxes para realizar la transposición del eje X e Y.
        tempImg = np.swapaxes(tempImg, 0, 1)
        tempImg = cv.copyMakeBorder(tempImg, vMid, vMid, 0, 0, borderType, value=color)

        # Ídem a lo anterior.
        for i in range(0, y):
            tots = np.sum(tempImg[i: i + lenV, vMid:x + vMid, :] * vMask, 0)
            convImg[:,i,:] = tots
    
    return convImg
        


def getDiff(imgA, imgB):
    """
    Obtener el error cuadrado medio de dos imágenes.

    Returns
    -------
    Flotante
        Error cuadrado medio de dos imágenes.

    """
    return np.mean(np.sqrt(pow(imgA - imgB, 2)))


imgMotorBike = leeImagen("./imagenes/motorcycle.bmp", False)

# Se elige una imagen, un sigma y las máscaras.
oImg = imgMotorBike
sigma = 2
mask = gaussianMask(0, sigma)
deriv = gaussianMask(1, sigma)

# Se desenfoca con la función implementada.
imgConv = convolveImage(oImg, mask, mask, cv.BORDER_REPLICATE)

# Se compara con el desenfoque que trae OpenCV.
imgCVConv = cv.GaussianBlur(oImg, (13,13), sigma, borderType=cv.BORDER_REPLICATE)
imgCVConv2 = cv.GaussianBlur(oImg, (0,0), sigma, borderType=cv.BORDER_REPLICATE)

# Mostrando las imágenes:
pintaI(oImg, "Imagen original")
pintaI(imgConv, "Implementación")
pintaI(imgCVConv, "OpenCV")

print("La diferencia media es:")
print("Si se indica tanto sigma como el tamaño de la máscara a OpenCV:", getDiff(imgConv, imgCVConv))
print("Si se indica solo sigma:", getDiff(imgConv, imgCVConv2))


# Generando 3 máscaras con longitud 5, 7 y 9 con la función del ejercicio A.
gaussMask5 = gaussianMask(0, None, 5)
gaussMask7 = gaussianMask(0, None, 7)
gaussMask9 = gaussianMask(0, None, 9)

gaussDeri5 = gaussianMask(1, None, 5)
gaussDeri7 = gaussianMask(1, None, 7)
gaussDeri9 = gaussianMask(1, None, 9)

# Generando imágenes de derivadas con esas máscaras:
imgdx = convolveImage(oImg, gaussDeri5, gaussMask5)
imgdy = convolveImage(oImg, gaussMask5, gaussDeri5)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy}, "Derivada por Ejes, T=5")
pintaI(mixed, "Magnitud de Gradiente")

imgdx = convolveImage(oImg, gaussDeri7, gaussMask7)
imgdy = convolveImage(oImg, gaussMask7, gaussDeri7)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy}, "Derivada por Ejes, T=7")
pintaI(mixed, "Magnitud de Gradiente")


imgdy = convolveImage(oImg, gaussMask9, gaussDeri9)
imgdx = convolveImage(oImg, gaussDeri9, gaussMask9)
mixed = np.sqrt(pow(imgdx, 2) + pow(imgdy, 2))
pintaIMVentana({"Derivada en X" : imgdx, "Derivada en Y" : imgdy}, "Derivada por Ejes, T=9")
pintaI(mixed, "Magnitud de Gradiente")


#%%
# Ejercicio 1D

print("\n---Ejercicio 1D---")

def laplacianMask(sigma = None, maskSize = None):
    """
    Obtener una máscara laplaciana 2D.
    Se debe pasar o el sigma o el maskSize.

    Parameters
    ----------
    sigma : Flotante, optional
        Sigma con el que generar las máscaras gaussianas.
    maskSize : TYPE, optional
        Tamaño de máscara con el que generar máscaras gaussianas.

    Returns
    -------
    L : Matriz Numpy
        Máscara Laplaciana 2D

    """
    # Obtener las máscaras: de alisamiento y de derivada segunda.
    gauss = gaussianMask(0, sigma, maskSize)
    gdxx = gaussianMask(2, sigma, maskSize)
    
    # Realizar el producto matricial para obtener las máscaras 2D de derivada por cada eje.
    dxx = np.outer(gauss, gdxx)
    dyy = np.outer(gdxx, gauss)
    
    # Si se paso la longitud de máscara, obtener el sigma.
    if(sigma == None):
        sigma = (maskSize - 1) / 6 
    
    # Para normalizar se multiplica por sigma al cuadrado.
    L = pow(sigma, 2) * (dxx + dyy)
    
    return L

def laplacian(img, sigma = None, maskSize = None, borderType = cv.BORDER_REFLECT):
    """
    Obtener la Laplaciana de Gaussiana de una imagen

    Parameters
    ----------
    img : Imagen
    sigma : Flotante, optional
        Sigma con el que generar las máscaras gaussianas.
    maskSize : TYPE, optional
        Tamaño de máscara con el que generar máscaras gaussianas.
    borderType : Tipo de borde de OpenCV, optional
        Borde para la convolución. Por defecto es cv.BORDER_REFLECT.

    Returns
    -------
    L : Imagen
        Imagen con la Laplaciana de Gaussiana aplicada.

    """
    # Obtener las máscaras
    gauss = gaussianMask(0, sigma, maskSize)
    gdxx = gaussianMask(2, sigma, maskSize)
    
    # Obtener la derivada de la imagen por X e Y
    dxx = convolveImage(img, gdxx, gauss, borderType)
    dyy = convolveImage(img, gauss, gdxx, borderType)
    
    # Normalizarlo
    dxx = normalize(dxx)
    dyy = normalize(dyy)
    
    if(sigma == None):
        sigma = (maskSize - 1) / 6 
    
    # Normalizarlo en el sentido de que el zero-crossing quede en los bordes.
    L = pow(sigma, 2) * (dxx + dyy)
    return L
    

def paintLapMask(lapMask, title=None):
    """
    Función auxiliar para imprimir la máscara gaussiana como una superifice 3D

    """
    ax = plt.axes(projection='3d')
    plt.title(title)

    (x, y) = np.meshgrid(np.arange(lapMask.shape[0]), np.arange(lapMask.shape[1]))

    ax.plot_surface(x, y, lapMask, cmap=plt.cm.coolwarm)
    plt.show()


imgCat = leeImagen("./imagenes/cat.bmp", False)

lapImage = imgCat
lapSigma1 = 1
lapSigma3 = 3

print("Filtrando imagen con Laplacianas...")
lapResult1 = laplacian(lapImage, lapSigma1)
lapResultCV1 = cv.Laplacian(lapImage, cv.CV_64F, ksize = 7, borderType=cv.BORDER_REFLECT)
print("Listo.")

print("Pintando imágenes...")
pintaI(lapImage, "Imagen Original")
pintaI(lapResult1, "Laplaciano: Implementación (σ=" + str(lapSigma1)+")")
pintaI(lapResultCV1, "Laplaciano: OpenCV")
print("Listo.")


lapResult3 = laplacian(lapImage, lapSigma3)
lapResultCV3 = cv.Laplacian(lapImage, cv.CV_64F, ksize = 19)

pintaI(lapResult3, "Laplaciano: Implementación (σ=" + str(lapSigma3)+")")
pintaI(lapResultCV3, "Laplaciano: OpenCV")

print("Generando solo máscaras Laplacianas...")
lapMask1 = laplacianMask(lapSigma1, None)
lapMask3 = laplacianMask(lapSigma3, None)
print("Listo.")

paintLapMask(lapMask1, "Máscara Laplaciana (σ=" + str(lapSigma1)+")")
paintLapMask(lapMask3, "Máscara Laplaciana (σ=" + str(lapSigma3)+")")

#%%
# Ejercicio 2A

print("\n---Ejercicio 2A---")


def subSample(img):
    """
    Reducir a la mitad una imagen

    """
    result = img[::2, ::2]
    return result

def gaussianPyramid(img, maxLevel, sigma=1, maskSize=None, borderType=cv.BORDER_REFLECT):
    """
    Generar la pirámide gaussiana de una imagen.

    Parameters
    ----------
    img : Imagen
    maxLevel : Entero
        Altura de la pirámide
    sigma : Flotante, optional
        Sigma con el que realizar el desenfoque. Por defecto es 1.
    maskSize : Entero
        Tamaño de máscara, usado cuando no se provee sigma. Por defecto es None.
    borderType : Tipo de Borde de OpenCV, opcional.
        Borde a utilizar en convolución. Por defecto es cv.BORDER_REFLECT.

    Returns
    -------
    imgList : Lista de imágenes
        Lista de imágenes que contiene la pirámide Gaussiana.

    """        
    # Comprobación de que se pasa un nivel mayor de 0, ¿es util? ¡Bueno...! ¡Ya lo escribí!
    if (maxLevel > 0):
        # Obtener la máscara
        mask = gaussianMask(0, sigma, maskSize)
        # El nivel 0 es la imagen original.
        imgList = [img]
        
        # Resto de niveles generados convolucionando y luego achicando la imagen del nivel actual.
        # En el caso del primer nivel es la imagen de entrada.
        for i in range(0, maxLevel):
            tempI = convolveImage(img, mask, mask, borderType)
            img = subSample(tempI)
            imgList.append(img)
    else:
         imgList = None       
         
    return imgList

def gaussianPyrCV(img, maxLevel, p_borderType=cv.BORDER_REFLECT):
    """
    Generar pirámide Gaussiana por medio de OpenCV
    Parameters
    ----------
    img : Imagen
        Imagen a la que obtener la pirámide Gaussiana.
    maxLevel : Entero
        Tamaño de la pirámide
    p_borderType : Tipo de Borde de OpenCV, opcional.
        Borde a utilizar en convolución. Por defecto es cv.BORDER_REFLECT.

    Returns
    -------
    imgList : Lista de imágenes
        Lista de imágenes que contiene la pirámide Gaussiana.

    """
    # pyrDown se encarga de la mayoría de las cosas implementadas antes.
    imgList = [img]
    for i in range(0, maxLevel):
        tempI = cv.pyrDown(img, borderType=p_borderType)
        img = tempI
        imgList.append(img)
        
    return imgList


def gaussCompare(pyr1, pyr2):
    """
    Comparar la diferencia de errores cuadrados en la pirámide.
    """
    size = len(pyr1)
    diff = 0
    for i in range(1, size):
        diff += getDiff(pyr1[i], pyr2[i])
    diff = diff / (size-1)
    return diff
    
imgBike = leeImagen("./imagenes/bicycle.bmp", False)

baseImg = imgBike
print("Generando pirámides gaussianas...")
gaussPyr06T = gaussianPyramid(baseImg, 4, None, 5)
gaussPyr06S = gaussianPyramid(baseImg, 4, 0.6)
gaussPyr2 = gaussianPyramid(baseImg, 4, 2)
gaussPyrLst = gaussianPyramid(baseImg, 4, 1)
gaussPyrCV = gaussianPyrCV(baseImg, 4)
print("Listo.")


print("Pintando pirámides gaussianas...")
pintaIM(gaussPyrLst, title="Implentación (σ=1)")
pintaIM(gaussPyr06S, title="Implentación (σ=0.6)")
pintaIM(gaussPyr06T, title="Implentación (T=5)")
pintaIM(gaussPyr2, title="Implentación (σ=2)")
print("Listo.")

pintaIM(gaussPyrCV,title="OpenCV")

print("Comparando diferencias entre pirámides: ")
print("\tCV vs σ=1:",gaussCompare(gaussPyrCV, gaussPyrLst))
print("\tCV vs σ=0.6:",gaussCompare(gaussPyrCV, gaussPyr06S))
print("\tCV vs T=5:",gaussCompare(gaussPyrCV, gaussPyr06T))
print("\tCV vs σ=2:",gaussCompare(gaussPyrCV, gaussPyr2))


#%%
# Ejercicio 2B

print("\n---Ejercicio 2B---")

def laplacianPyramid(p_gaussPyr):
    """
    Generar pirámide Laplaciana a partir de pirámide Gaussiana

    Parameters
    ----------
    p_gaussPyr : Lista de imágenes
        Vector que contiene la pirámide gaussiana.

    Returns
    -------
    imgList : Lista de imágenes
        Vector que contiene la pirámide laplaciana.

    """
    # Invertir el vector, por comodidad.
    gaussPyr = p_gaussPyr[::-1]
    # Obtener longitud.
    maxLevel = len(gaussPyr)
    # Preparar vector.
    imgList = []
    imgList.append(gaussPyr[0])

    # Por cada imagen, expandirla y restarle la imagen que le precede en la pirámide gaussiana hasta llegar al nivel 0.
    for i in range(1, maxLevel):
        expandedImg = cv.resize(gaussPyr[i-1], (gaussPyr[i].shape[1], gaussPyr[i].shape[0]), interpolation=cv.INTER_LINEAR)
        tempI = gaussPyr[i] - expandedImg
        imgList.append(tempI)
    
    return imgList
    
def laplacianPyrCV(p_gaussPyr):
    gaussPyr = p_gaussPyr[::-1]
    maxLevel = len(gaussPyr)
    imgList = []
    imgList.append(gaussPyr[0])
    
    for i in range(1, maxLevel):
        expandedImg = cv.pyrUp(gaussPyr[i-1], dstsize=(gaussPyr[i].shape[1], gaussPyr[i].shape[0]))
        tempI = gaussPyr[i] - expandedImg
        imgList.append(tempI)
    
    return imgList

print("Generando pirámides Laplacianas...")
lapPyrLst = laplacianPyramid(gaussPyrLst)
lapPyrCV = laplacianPyrCV(gaussPyrLst)
print("Listo.")

print("Pintando pirámides...")
pintaIM(lapPyrLst, "Implementación")
pintaIM(lapPyrCV, "OpenCV")
print("Listo.")


#%%
# Exercise 2C
print("\n---Ejercicio 2C---")

def recoverImg(lapPyr):
    """
    Recuperar una imagen con la pirámide Laplaciana.

    Parameters
    ----------
    lapPyr : Lista de Imágenes
        Pirámide laplaciana como un vector de imágenes

    Returns
    -------
    baseImg : Imagen
        Imagen recuperada por la pirámide.

    """
    # Obtener el nivel de la pirámide.
    maxLevel = len(lapPyr)
    # Obtener la base
    baseImg = lapPyr[0]
    
    for i in range(1, maxLevel):
        # Expandir la imagen y sumarla con la imagen siguiente de la pirámide y repetir el proceso.
        expandedImg = cv.resize(baseImg, (lapPyr[i].shape[1], lapPyr[i].shape[0]), interpolation=cv.INTER_LINEAR)
        baseImg = lapPyr[i] + expandedImg
         
    return baseImg

print("Reconstruyendo Imagen...")
recoveredImg = recoverImg(lapPyrLst)
print("Listo.")

print("Pintando comparación...")
pintaIMVentana({"Imagen Original" : baseImg, "Imagen Recuperada" : recoveredImg}, "Reconstrucción")
print("Listo.")

print("La diferencia de la imagen original y la recuperada es", getDiff(baseImg, recoveredImg))
    
#%%
# Bonus 1

print("\n---Ejercicio Bonus 1---")

def genHybridImg(highF, hFSigma, lowF, lFSigma):
    """
    Generar una imagen híbrida

    Parameters
    ----------
    highF : Imagen
        Imagen de alta frecuencia.
    hFSigma : Flotante
        Sigma para la imagen de alta frecuencia.
    lowF : Imagen
        Imagen de baja frecuencia.
    lFSigma : Foltante
        Sigma para la imagen de baja frecuencia.

    Returns
    -------
    hybrid : Imagen
        La imagen híbrida.
    imgDict : Diccionario de Imágenes
        Contiene la imagenes de alta y baja frecuencia filtradas junto a la imagen híbrida.

    """
    # Se genera la máscara de la imagen de baja frecuencia con el sigma.
    lowMask = gaussianMask(0, lFSigma, None)    
    
    # Se genera la máscara de alta frecuencia con la Laplaciana
    # *-1 pues invertir los colores hace que se mezcle mejor las imágenes.
    highPass = laplacian(highF, hFSigma) * -1
    
    # Se convoluciona para obtener la imagen desenfocada.
    lowPass = convolveImage(lowF, lowMask, lowMask)

    # Se normalizan los valores para luego sumarlos correctamente.
    highPass = normalize(highPass)
    lowPass = normalize(lowPass)
    
    hybrid = highPass + lowPass
        
    imgDict = {"Alta Frecuencia" : highPass, "Baja Frecuencia" : lowPass, "Híbrida" : hybrid}
    return hybrid, imgDict
    

imgDog = leeImagen("./imagenes/dog.bmp", False)
imgBird = leeImagen("./imagenes/bird.bmp", False)
imgPlane = leeImagen("./imagenes/plane.bmp", False)

hyb, imgD = genHybridImg(imgBike, 1.5, imgMotorBike, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD, "Bici-Moto")
pintaIM(imgPyr)

hyb, imgD = genHybridImg(imgBird, 1.5, imgPlane, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD, "Pájaro-Avión")
pintaIM(imgPyr)

hyb, imgD = genHybridImg(imgCat, 2, imgDog, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD, "Gato-Perro")
pintaIM(imgPyr)


#%%
# Bonus 2

print("\n---Ejercicio Bonus 2---")

imgCatRGB = leeImagen("./imagenes/cat.bmp", True)
imgDogRGB = leeImagen("./imagenes/dog.bmp", True)

imgBirdRGB = leeImagen("./imagenes/bird.bmp", True)
imgPlaneRGB = leeImagen("./imagenes/plane.bmp", True)

imgMotorBikeRGB = leeImagen("./imagenes/motorcycle.bmp", True)
imgBikeRGB = leeImagen("./imagenes/bicycle.bmp", True)


hyb, imgD = genHybridImg(imgBikeRGB, 1.5, imgMotorBikeRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr, "Moto-Bicicleta", (1,1,1))


hyb, imgD = genHybridImg(imgBirdRGB, 1.5, imgPlaneRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr, "Pájaro-Avión", (1,1,1))


hyb, imgD = genHybridImg(imgCatRGB, 2, imgDogRGB, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr, "Gato-Perro", (1,1,1))


#%%
# Bonus 3

print("\n---Ejercicio Bonus 3---")

imgModelT = leeImagen("./imagenes/modelt.bmp", True)
imgTeslaX = leeImagen("./imagenes/tesla.bmp", True)

hyb, imgD = genHybridImg(imgModelT, 2, imgTeslaX, 9)
imgPyr = gaussianPyramid(hyb, 4)
pintaIMVentana(imgD)
pintaIM(imgPyr, "Pasado a Futuro",(1,1,1))
