#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 2: Detección de Puntos Relevantes y Construcción de Panoramas
    Asignatura: Vision por Computador
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Noviembe 2021
    
[ENGLISH]

    Practice 1: Detecting Relevant Points and Panorama Construction
    Course: Computer Vision
    Author: Valentino Lugli (Github: @RhinoBlindado)
    November 2021

"""

# LIBRERÍAS

#   Usando Matplotlib para mostrar imágenes
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#   Usando OpenCV para el resto de cosas con imágenes
import cv2 as cv

#   Usando Numpy para cálculos matriciales
import numpy as np

#   Usando Math para funciones matemáticas avanzadas
import math

#   Usando Random para obtener puntos
import random

from numba import jit

# FUNCIONES AUXILIARES
# - Provienen de prácticas anteriores.

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


def addPadding(img, sizePadding, typePadding, color=None):

    if(cv.BORDER_CONSTANT):
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding, value=color)
    else:
        paddedImg = cv.copyMakeBorder(img, sizePadding, sizePadding, sizePadding, sizePadding, typePadding)

    return paddedImg

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
        

def getEuclidDiff(imgA, imgB):
    """
    Obtener el error euclideo medio de dos imágenes.

    Returns
    -------
    Flotante
        Error cuadrado medio de dos imágenes.

    """
    return np.mean(np.sqrt(pow(imgA - imgB, 2)))


def subSample(img):
    """
    Reducir a la mitad una imagen

    """
    result = img[::2, ::2]
    return result

def wait():
    input("Pulsa una tecla para continuar")

############## FIN FUNCIONES AUXILIARES ##############

############## INICIO FUNCIONES DE LA PRACTICA ACTUAL ##############
def getSigmaOct(sigma0, s, ns):
    return sigma0 * math.sqrt( pow(2, (2 * s) / ns) - pow(2, 2 * (s - 1) / ns) )

def genOctaves(p_img, scales, sigma0, extra = 3):
    
    octaves = []
    diffOfGauss = []
    
    actSigma = sigma0
    actImg = p_img
    octNum = scales + extra
    
    octaves.append(actImg)
    
    for i in range(1, octNum):
        mask  = gaussianMask(0, actSigma, None)
        octav = convolveImage(actImg, mask, mask)
        
        octaves.append(octav)
        actSigma = getSigmaOct(sigma0, i, scales)

        diffOfGauss.append(actImg - octav)
        
        actImg = octav

    return octaves, diffOfGauss


def isLocalExtrema(x, y, localLayer, backLayer, frontLayer): 
    
    actVal      = np.abs(localLayer[x][y])
    layerVal    = np.abs(localLayer[x-1:x+2, y-1:y+2]).max()
    backVal     = np.abs(backLayer [x-1:x+2, y-1:y+2]).max()
    frontVal    = np.abs(frontLayer[x-1:x+2, y-1:y+2]).max()
        
    return (actVal == max(actVal, layerVal, backVal, frontVal))
    
def getRealSigma(k, scale, sigma_0=1.6):
    
    shift = (scale - 1) * 3
    
    return (sigma_0 * pow(2, (shift+k)/3))

def realCoords(i, l):
    
    if(l > 1):
        x = round (i * 2 * (l-1))
    elif(l == 1):
        x = i
    else:
        x = round( i / 2 )
    
    return x

def getExtrema(DoG):
    
    keyPLst = []
    keyCV = []
    
    for l, octave in enumerate(DoG):
        for k in range(1, len(octave)-1):
            for i in range(1, octave[k].shape[0] - 1):
                for j in range(1, octave[k].shape[1] - 1):
                    if(isLocalExtrema(i , j, octave[k], octave[k-1], octave[k+1])):
                        keyPLst.append((np.abs(octave[k][i][j]), i, j, l, getRealSigma(k, l)))

    keyPLst.sort(key=lambda tup: tup[0], reverse=True)
    keyBest = keyPLst[0:250]
    
    for i in keyBest:
        keyCV.append(cv.KeyPoint(realCoords(i[2], i[3]), realCoords(i[1], i[3]), round(i[4]) * 12, response=i[0], octave=i[3]))
    
    return keyBest, keyCV, keyPLst
    
def siftDetector(p_img, numOctaves, numScales, sigma0):
    
    # Escalando
    img = cv.resize(p_img, (p_img.shape[1]*2, p_img.shape[0]*2), interpolation=cv.INTER_LINEAR)
    
    siftPyr = []
    siftDoG = []
    # Generando piramide
    for i in range(0, numOctaves + 1):
        octave, DoG = genOctaves(img, numScales, sigma0)
        
        img = subSample(octave[numScales])
        siftPyr.append(octave)
        siftDoG.append(DoG)
    
    return siftPyr, siftDoG

#%%
yose1 = leeImagen("./imagenes/Yosemite1.jpg", False)
yose1 = yose1.astype(np.uint8)

gaussPyr, DoGPyr = siftDetector(yose1, 3, 3, 1.6)

keyPoints, kpCV, kpAll = getExtrema(DoGPyr)

yoseOut = cv.drawKeypoints(yose1, kpCV, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
pintaI(yoseOut)
#%%

def getKeyPoints_BF(img1, img2):
    sift = cv.SIFT_create()
    
    # Objeto para hacer BruteForce + CrossCheck; para eso se habilita con true en el
    # segundo parámetro y normalización L1 que se recomienda para SIFT.
    bfCross = cv.BFMatcher_create(cv.NORM_L1, True)

    # Obteniendo caracterisitcas: keypoints y descriptores en un solo paso.
    #   - Las imágenes tienen que convertirse a uint8, es lo que acepta OpenCV,
    #     sino se queja.
    
    img1K, img1D = sift.detectAndCompute(img1, None)
    img2K, img2D = sift.detectAndCompute(img2, None)

    # Haciendo el matching de Bruteforce+CrossCheck
    bfCrossRes = bfCross.match(img1D, img2D)

    return bfCrossRes, img1K, img2K
    

def getKeyPoints_Lowe2NN(img1, img2, p_k=2):
    sift = cv.SIFT_create()
    
    # Objeto para hacer Promedio de Lowe-2nn, sin CrossCheck (por defecto)
    bfLowe2nn = cv.BFMatcher_create(cv.NORM_L1)

    # Obteniendo caracterisitcas: keypoints y descriptores en un solo paso.
    #   - Las imágenes tienen que convertirse a uint8, es lo que acepta OpenCV,
    #     sino se queja.
    
    img1K, img1D = sift.detectAndCompute(img1, None)
    img2K, img2D = sift.detectAndCompute(img2, None)
    
    # Haciendo matching, primero con k-vecinos cercanos; k=2 (por defecto)
    loweMatch = bfLowe2nn.knnMatch(img1D, img2D, k=p_k)

    bfLoweBest = []
    # Se descartan los puntos ambiguos según el criterio de Lowe
    for i, j in loweMatch:
        if i.distance < 0.8*j.distance:
            bfLoweBest.append([i])
            
    return bfLoweBest, img1K, img2K

def ejercicio2():
    #  "Con cada dos de las imágenes de Yosemite con solapamiento detectar y  
    #   extraer los descriptores SIFT de OpenCV, usando para ello la función 
    #   cv2.detectAndCompute()..."
    
    yose1 = leeImagen("./imagenes/Yosemite1.jpg", False).astype('uint8')
    yose2 = leeImagen("./imagenes/Yosemite2.jpg", False).astype('uint8')
    
    bfCrossRes, yose1Keys, yose2Keys = getKeyPoints_BF(yose1, yose2)
    bfLoweBest, a, b = getKeyPoints_Lowe2NN(yose1, yose2)    
    
    #   "... mostrar ambas imágenes en un mismo canvas y pintar líneas de diferentes 
    #   colores entre las coordenadas de los puntos en correspondencias. Mostrar 
    #   en cada caso un máximo de 100 elegidas aleatoriamente."
    
    # Obtener 100 puntos aleatorios (o si hay menos de 100 solo esos puntos)
    crossPoints = random.sample(bfCrossRes, min(100, len(bfCrossRes)))
    lowePoints = random.sample(bfLoweBest, min(100, len(bfLoweBest)))
    
    # Obteniendo las líneas de colores entre cada imagen dado los keypoints y descriptores.
    bfCrossRes = cv.drawMatches(yose1, yose1Keys, yose2, yose2Keys, crossPoints, None,flags=2)
    bfLoweRes = cv.drawMatchesKnn(yose1, yose1Keys, yose2 , yose2Keys, lowePoints, None,flags=2)
    
    # Pintar las imágenes.
    pintaI(bfCrossRes, "SIFT: Bruteforce + CrossCheck")
    pintaI(bfLoweRes, "SIFT: Lowe-Average-2NN")


def ejercicio3():
    
    left = leeImagen("./imagenes/IMG_20211030_110413_S.jpg", True)
    center = leeImagen("./imagenes/IMG_20211030_110415_S.jpg", True)
    right = leeImagen("./imagenes/IMG_20211030_110417_S.jpg", True)
    
    right = right.astype(np.uint8)
    center = center.astype(np.uint8)
    left = left.astype(np.uint8)
    
    leftCenter, k1, k2 = getKeyPoints_Lowe2NN(left, center)
    
    src_pts = np.float32([ k1[m[0].queryIdx].pt for m in leftCenter ]).reshape(-1,1,2)
    dst_pts = np.float32([ k2[m[0].trainIdx].pt for m in leftCenter ]).reshape(-1,1,2)
    
    canvas = np.zeros((500, 800))
    baseHom = np.array([[1, 0, canvas.shape[1]/2 - center.shape[1]/2],
                     [0, 1, canvas.shape[0]/2 - center.shape[0]/2],
                     [0, 0, 1]],
                    dtype=np.float64)
    
    # canvas = cv.warpPerspective(center, baseHom, (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT)
    
    homM, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1)
    
    centerRight, kl1, kl2 = getKeyPoints_Lowe2NN(right, center)
    
    src_pts = np.float32([ kl1[m[0].queryIdx].pt for m in centerRight ]).reshape(-1,1,2)
    dst_pts = np.float32([ kl2[m[0].trainIdx].pt for m in centerRight ]).reshape(-1,1,2)
    
    homL, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1)
    
    
    canvas = cv.warpPerspective(right, baseHom.dot(homL), (canvas.shape[1], canvas.shape[0]), dst=canvas) 
    canvas = cv.warpPerspective(left, baseHom.dot(homM), (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT) 
    canvas = cv.warpPerspective(center, baseHom, (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT)
    pintaI(canvas)


#%%
print("Ejercicio 1: ")
ejercicio1()
#%%
print("Ejercicio 2: ")
ejercicio2()
#%%
print("Ejercicio 3: ")
ejercicio3()

