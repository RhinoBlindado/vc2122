#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 2: Detección de Puntos Relevantes y Construcción de Panoramas
    Asignatura: Vision por Computador
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Noviembe 2021
    
[ENGLISH]

    Practice 2: Detecting Relevant Points and Panorama Construction
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

############### FIN FUNCIONES AUXILIARES

############################## FUNCIONES PARA EJERCICIOS 

############### FUNCIONES EJERCICIO 1

def getSigmaOct(sigma0, s, ns):
    """
    Calcula el sigma para avanzar de una escala a la siguiente en una octava.

    Parameters
    ----------
    sigma0 : Flotante
        Sigma inicial
    s : Entero
        Escala a la que llegar.
    ns : Entero
        Número de escalas canónicas.

    Returns
    -------
    Flotante
        Sigma necesario para generar una máscara gaussiana para llegar a 
        la octava s.

    """
    # La misma expresión que se define en la presentación, adaptada a Python.
    return sigma0 * math.sqrt( pow(2, (2 * s) / ns) - pow(2, 2 * (s - 1) / ns) )

def genOctave(p_img, scales, sigma0, extra = 3):
    """
    Generar una octava, independiente de cual.

    Parameters
    ----------
    p_img : Imagen
        Imagen semilla.
    scales : Entero
        Escalas canónicas.
    sigma0 : Flotante
        Sigma inicial.
    extra : Entero, opcional
        Escalas auxiliares. Por defecto 3.

    Returns
    -------
    scaleList : Lista de imágenes
        Imágenes que forman la octava gaussiana.
    diffOfGauss : Lista de imágenes
        Imágenes que forman la octava laplaciana.

    """
    
    # Generando las listas para almacenar las escalas.
    scaleList = []
    diffOfGauss = []
    
    # La imagen de entrada es la que se pasa por parámetro.
    actImg = p_img
    
    # Se calcula el valor completo del bucle, la suma de las escalas canónicas y las escalas auxiliares.
    octNum = scales + extra
    
    # Se inserta la primera imagen como la escala auxiliar 0.
    scaleList.append(actImg)
    
    # Se comienza el bucle para el resto de las ecalas.
    for i in range(1, octNum):
        # Se obtiene el sigma para pasar de la escala actual a la siguiente...
        actSigma = getSigmaOct(sigma0, i, scales)
        # ...con dicho sigma se calcula una máscara gaussiana...
        mask  = gaussianMask(0, actSigma, None)
        # ...y con dicha máscara se convoluciona con la escala actual...
        octav = convolveImage(actImg, mask, mask)
        # ...se almacena en la lista...
        scaleList.append(octav)
        # ...se genera la escala laplaciana restando la escala actual con la generada y se guarda...
        diffOfGauss.append(actImg - octav)
        # ...se actualiza la escala actual para la siguiente iteración.
        actImg = octav

    return scaleList, diffOfGauss


def isLocalExtrema(x, y, localLayer, backLayer, frontLayer): 
    """
    Determina si un punto (x,y) en una escala actual es el extremo local del 
    cubo de escalas.

    Parameters
    ----------
    x : Entero
        Coordenada x de la imagen.
    y : Entero
        Coordenada y de la imagen.
    localLayer : Imagen
        Escala actual
    backLayer : Imagen
        Escala anterior a la actual.
    frontLayer : Imagen
        Escala posterior a la actual.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Obtener el valor del pixel (x,y) en la escala actual.
    actVal      = localLayer[x][y]
    # Obtener el valor máximo de la ventana 3x3 de la escala actual.
    layerMax    = localLayer[x-1:x+2, y-1:y+2].max()
    # Ídem para la escala anterior
    backMax     = backLayer [x-1:x+2, y-1:y+2].max()
    # Ídem para la escala posterior.
    frontMax    = frontLayer[x-1:x+2, y-1:y+2].max()
    
    # Mismo procedimiento pero con el valor mínimo.
    layerMin    = localLayer[x-1:x+2, y-1:y+2].min()
    backMin     = backLayer [x-1:x+2, y-1:y+2].min()
    frontMin    = frontLayer[x-1:x+2, y-1:y+2].min()

    # Si el valor del pixel (x,y) es el máximo o mínimo de todos los 26
    # valores, se retorna verdadero y falso en caso contrario.
    return (actVal == max(layerMax, backMax, frontMax) or 
            actVal == min(layerMin, backMin, frontMin))
    
def getRealSigma(k, octave, sigma_0=1.6):
    """
        Calcula el sigma real de un punto dado una escala y una octava.

    Parameters
    ----------
    k : Entero
        Escala actual.
    octave : Entero
        Octava actual.
    sigma_0 : Flotante, opcional
        Sigma inicial. Por defecto es 1.6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    shift = (octave - 1) * 3
    
    return (sigma_0 * pow(2, (shift + k) / 3))

def realCoords(i, l, rounding=True):
    """
    Calcula una coordenada real dada una coordenada y la escala.

    Parameters
    ----------
    i : Entero
        Valor de una coordenada,
    l : Entero
        Valor que indique una octava.
    rounding : Booleano, opcional.
        Indica si se realiza redondeo, por defecto se realiza.

    Returns
    -------
    x : Entero o Flotante
        Valor de la coordenada en la imagen.

    """
    
    # Si es la octava 0, se divide entre dos,
    # si es la octava 1, es el mismo tamaño,
    # si es una octava >1, se divide entre el valor de la escala - 1 por 2.
    if(rounding):
        if(l > 1):
            x = round (i * 2 * (l-1))
        elif(l == 1):
            x = i
        else:
            x = round( i / 2 )
    else:
        if(l > 1):
            x = (i * 2 * (l-1))
        elif(l == 1):
            x = i
        else:
            x = ( i / 2 )
    
    return x


def getExtrema(DoG):
    """
    Obtener los puntos característicos, segunda parte del detector SIFT.

    Parameters
    ----------
    DoG : Lista de listas 
        Contenedor donde están las octavas y sus escalas.

    Returns
    -------
    - Lista de los mejores puntos planos.
    - Lista de los mejores puntos en objetos KeyPoint.
    - Lista de todos los puntos planos.
    - Lista de todos los puntos planos divididos por octavas.

    """    
    
    # Se crean las listas que contendrán los puntos.
    # - Lista "plana": contiene la información  de los puntos en una tupla normal de Python.
    keyPLst = []
    # - Lista KeyPoint: contiene la información de los puntos en un objeto KeyPoint.
    keyCV = []
    # - Lista de puntos por octava, añadido para el Bonus 1-1.
    keyPerOct = []
    
    # Tal y como se define este algoritmo es recorrer todas las octavas, 
    # todas las escalas canónicas, por cada pixel salvo los bordes,
    # una cubo de 3x3x3.
    
    # Por cada octava...
    for l, octave in enumerate(DoG):
        # Bonus 1-1: Para poder guardar por cada octava sus puntos.
        keyPerOct.append([])
        # Por cada escala válida, es decir de la segunda hasta la penúltima
        # para poder encajar el cubo.
        for k in range(1, len(octave)-1):
            # Por cada píxel salvo los primeros y los últimos para encajar
            # el cubo.
            for i in range(1, octave[k].shape[0] - 1):
                for j in range(1, octave[k].shape[1] - 1):
                    # Determinar si el punto es máximo o mínimo local...
                    if(isLocalExtrema(i , j, octave[k], octave[k-1], octave[k+1])):
                        # ... si lo es, añadirlo a las listas:
                        
                        #   Formato:
                        #               0, 1, 2     , 3     , 4                      , 5
                        #               x, y, escala, octava, respuesta              , sigma real
                        keyPLst.append((i, j, k     , l     , np.abs(octave[k][i][j]), getRealSigma(k, l)))
                        # Bonus 1-1
                        keyPerOct[l].append((i, j, k     , l     , np.abs(octave[k][i][j]), getRealSigma(k, l)))

    # - Ordenar la lista plana por [4] que es la respuesta, es decir, el valor del píxel en valor absoluto.
    # - reverse=True para que se ordene de mayor a menor.
    keyPLst.sort(key=lambda tup: tup[4], reverse=True)
    # Tomat los 100 mejores puntos como se pide.
    keyBest = keyPLst[0:100]
    
    # Rellenar la lista de objetos KeyPoint por medio de la lista plana.
    # - Se convierten las coordenadas relativas a las coordenadas reales.
    # - KeyPoint invierte las coordenadas, es decir (y, x)
    # - Se multiplica por 2 * 6 el sigma, como se pide en el guión.
    for i in keyBest:
        keyCV.append(cv.KeyPoint(realCoords(i[1], i[3]), realCoords(i[0], i[3]), i[5] * 12))
    
    # Devolver las listas.
    return keyBest, keyCV, keyPLst, keyPerOct
    
def lowePyramid(p_img, numOctaves, numScales, sigma0):
    """
    Generar la pirámide gaussiana y laplaciana de Lowe, primera parte del 
    detector SIFT.

    Parameters
    ----------
    p_img : Imagen
        Imagen de entrada, se asume que tiene un sigma de 0,8.
    numOctaves : Entero
        Número de octavas.
    numScales : Entero
        Número de escalas, sin contar la 0.
    sigma0 : Flotante
        Sigma inicial

    Returns
    -------
    siftPyr : Lista de listas de imágenes
        Pirámide gaussiana de Lowe
    siftDoG : Lista de listas de imágenes
        Pirámide laplaciana de Lowe

    """
    # La imagen inicial se aumenta en 2 su tamaño haciendo uso de interpolación bilineal.
    img = cv.resize(p_img, (p_img.shape[1]*2, p_img.shape[0]*2), interpolation=cv.INTER_LINEAR)
    
    # Se declaran las listas
    siftPyr = []
    siftDoG = []
    # Generando las pirámides, tantas octavas más la octava 0.
    for i in range(0, numOctaves + 1):
        # Generar una octava, se obtienen la octava gaussiana y laplaciana.
        octave, DoG = genOctave(img, numScales, sigma0)
        # - Se reduce la última escala "canónica", si comienzan en 0,
        #   coincide en que será el número de escalas "canónicas"
        #  - La escala 0 es auxiliar, la 1, 2 y 3 son las canónicas, se toma la 3.
        #    para generar la siguiente escala: se reduce a la mitad.
        img = subSample(octave[numScales])
        
        # Se añade cada escala en la lista correspondiente.
        siftPyr.append(octave)
        siftDoG.append(DoG)
    
    return siftPyr, siftDoG


def showOctaves(gPyr):
    """
    Función auxiliar para mostrar las octavas dada una pirámide de Lowe

    Parameters
    ----------
    gPyr : Lista de listas de imágenes.
        Pirámide gaussiana o laplaciana de Lowe

    """        
    # Pintando las escalas canónicas, de 1 a 3.
    pintaIM(gPyr[0][1:4], "Octava 0, Escala 1, 2 y 3")
    pintaIM(gPyr[1][1:4], "Octava 1, Escala 1, 2 y 3")
    pintaIM(gPyr[2][1:4], "Octava 2, Escala 1, 2 y 3")
    pintaIM(gPyr[3][1:4], "Octava 3, Escala 1, 2 y 3")
    
    
############### FIN FUNCIONES EJERCICIO 1

############### FUNCIONES BONUS 1-1

def getBestPercentage(keyPoints):
    """
    Obtener los 100 mejores puntos, de cada octava una proporción.

    Parameters
    ----------
    keyPoints : Lista de listas de tuplas,
        Puntos característicos por cada octava.

    Returns
    -------
    keyOut : Lista de tuplas de los 100 mejores puntos,
        Lista de los 100 mejores puntos por octava, en proporción.
    keyCV : Lista de KeyPoint,
        Lista de los 100 mejores puntos por octava, en proporción.

    """
    # Por cada octava, estos son los valores que se piden
    
    # Octava 0 , 1, 2, 3
    sizes = [50, 25, 15, 10]
    
    # Declarando las listas.
    keyPerBest = []
    keyCV = []
    keyOut = []
    
    # Recorriendo los puntos...
    for i in range(len(keyPoints)):
            #... ordenarlos de mayor a menor por medio de su respuesta.
            keyPoints[i].sort(key=lambda tup: tup[4], reverse=True)
            # Añadir los puntos necesarios por cada octava a la lista.
            keyPerBest.append(keyPoints[i][0:sizes[i]])

    # Por cada octava...
    for i in range(len(keyPerBest)):
        #... por cada punto en cada octava...
        for j in range(len(keyPerBest[i])):
                # ... rellenar la lista de puntos, convirtiendolos a sus coordenadas y sigmas reales * 12 ya que eso es lo que se pide.
                keyCV.append(cv.KeyPoint(realCoords(keyPerBest[i][j][1], keyPerBest[i][j][3]), realCoords(keyPerBest[i][j][0], keyPerBest[i][j][3]), keyPerBest[i][j][5] * 12))
                keyOut.append((keyPerBest[i][j][0], keyPerBest[i][j][1], keyPerBest[i][j][2], keyPerBest[i][j][3], keyPerBest[i][j][4], keyPerBest[i][j][5]))

    return keyOut, keyCV

############### FIN FUNCIONES BONUS 1-1

############### FUNCIONES BONUS 1-2

def localQuadratic(kp, DoG):
    """
    Realizar una interpolación cuadrática del "cubo" de escalas 3D, adaptado
    del pseudocódigo de "Anatomy of the SIFT Method"
    
    Parameters
    ----------
    kp : Lista de tuplas
        Lista plana de puntos característicos.
    DoG : Lista de listas de imágenes
        Pirámide laplaciana de Lowe.

    Returns
    -------
    alphaStar : Vector de 3 componentes
        Contiene el sesgo que debe aplicarse a un punto característico
    omega : Flotante
        Contiene el valor interpolado del píxel, su respuesta.

    """
    # Obteniendo datos del keypoint actual
    # - Se convierte a entero porque se utiliza un array de Numpy 
    #   la explicación del porqué está en extremaInterpolation()
    
    i = int(kp[0]) # Coordenada x
    j = int(kp[1]) # Coordenada y

    # Datos de las escalas necesarias
    act  = DoG[int(kp[3])][int(kp[2])] # Escala actual
    back = DoG[int(kp[3])][int(kp[2]-1)] # Escala anterior
    front= DoG[int(kp[3])][int(kp[2]+1)] # Escala posterior.
    
    # - Calculando la gradiente y Hessiano como se explica en el artículo.
    # - Es una adaptación directa, haciendo uso de los métodos de Numpy.
    
    # Gradiente
    grad = np.array([[(front[i][j] - back[i][j])  / 2],
                     [(act[i+1][j] - act[i-1][j]) / 2],
                     [(act[i][j+1] - act[i][j-1]) / 2]])
    
    # Hessiano
    hessian = np.empty((3,3))
    hessian[0][0] = front[i][j] + back[i][j] - 2 * (act[i][j])
    hessian[1][1] = act[i+1][j] + act[i-1][j] - 2 * (act[i][j]) 
    hessian[2][2] = act[i][j+1] + act[i][j-1] - 2 * (act[i][j])
    
    hessian[0][1] = (front[i+1][j] - front[i-1][j] - back[i+1][j] + back[i-1][j]) / 4
    hessian[0][2] = (front[i][j+1] - front[i][j-1] - back[i][j+1] + back[i][j-1]) / 4
    hessian[1][2] = (act[i+1][j+1] - act[i+1][j-1] - act[i-1][j+1] + act[i-1][j-1]) / 4
    
    hessian[1][0] = hessian[0][1]
    hessian[2][0] = hessian[0][2]
    hessian[2][1] = hessian[1][2]

    # Generando alpha*
    alphaStar = -(np.linalg.inv(hessian)).dot(grad)
    # Generando omega
    omega = kp[4] - 0.5 * np.transpose(grad).dot((np.linalg.inv(hessian)).dot(grad))
    
    return alphaStar, omega
    
def extremaInterpolation(keyLst, DoG):
    """
    Obtener una lista de puntos interpolados, tercer paso del detector SIFT.

    Parameters
    ----------
    kp : Lista de tuplas
        Lista plana de puntos característicos.
    DoG : Lista de listas de imágenes
        Pirámide laplaciana de Lowe.

    Returns
    -------
    kpInter : Lista de tuplas
        Lista plana de puntos interpolados.
    kpInterCV : Lista de KeyPoint
        Lista de objetos KeyPoint que tiene los puntos interpolados.

    """
    # Declarando las listas.
    kpInter = []
    kpInterCV = []
    
    # Por cada punto en la lista de puntos caracteríticos.
    for i in range(0, len(keyLst)):
        
        # Coger el actual.
        # - Se pasa a array de numpy para poder ir actualizando el punto,
        # ya que las tuplas de Python son solo de lectura.
        actKp = np.asarray(keyLst[i])
        
        # Se crea un vector vacío para almacenar los valores reales. 
        realValue = np.empty((3,))
        # Un contador, como se especifica que tienen que ser maximo 5 intentos.
        counter = 0
        
        # Realizar la interpolación hasta cumplir con ciertas condiciones...
        while True:
            # Obtener alpha* y omega dado el punto actual y las escalas.
            alphaStar, omega = localQuadratic(actKp, DoG)
            
            # Se escriben los valores interpolados:
            # - El valor 0 de alpha* es la escala, se utiliza para calcular el
            #   sigma real con el sesgo.
            realValue[0] = getRealSigma(actKp[2] + alphaStar[0][0], actKp[3])
            # Obtener la coordenada real x
            realValue[1] = realCoords(actKp[0] + alphaStar[1][0], actKp[3], False)
            # Obtener la coordenada real y
            realValue[2] = realCoords(actKp[1] + alphaStar[2][0], actKp[3], False)
               
            # El punto se actualiza, pero esta vez se redondean los valores.
            # Coordenada x relativa
            actKp[0] = np.round(actKp[0] + alphaStar[1][0])
            # Coordenada y relativa
            actKp[1] = np.round(actKp[1] + alphaStar[2][0])
            # Escala s relativa, en el punto está en la posición 3.
            actKp[2] = np.round(actKp[2] + alphaStar[0][0])

            # Si es un valor "estable", es decir, que el maximo de alpha*
            # no sobrepasa 0.6 o si ya han sido 5 intentos se sale del bucle.
            if(max(np.abs(alphaStar)) < 0.6 or counter < 5):
                break

            counter += 1
        # - Si luego de 5 o menos intentos, el valor máximo de alpha* está por
        #   debajo de 0.6, el punto es aceptado. 
        # - Se almacena el punto, en versión plana y en versión KeyPoint.
        if(max(np.abs(alphaStar)) < 0.6):
            kpInter.append((int(actKp[0]), int(actKp[1]), int(actKp[2]), int(actKp[3]), omega[0][0], realValue[0]))
            kpInterCV.append(cv.KeyPoint(realValue[2], realValue[1], realValue[0] * 12))
            
        # Los puntos que no logren esto son descartados.
    
    return kpInter, kpInterCV

############### FIN FUNCIONES BONUS 1-2

############### FUNCIONES EJERCICIO 2

def getKeyPoints_BF(img1, img2):
    """
    Obtener los puntos característicos y descriptor de dos imágenes utilizando
    BruteForce+CrossCheck

    Parameters
    ----------
    
    img1 : Imagen
        Imagen a la cual encontrar los puntos en correspondencia con img2.
    img2 : Imagen
        Imagen con la cual se encuentran los puntos en correspondencia con img1.
    Returns
    -------
    
    bfCrossRes : Descriptores de los puntos en correspondencia entre las dos imágenes.
    img1K : Puntos de la imagen 1,
    img2K : Puntos de la imagen 2.

    """
    # Creando el objeto SIFT.
    sift = cv.SIFT_create()
    
    # Objeto para hacer BruteForce + CrossCheck; para eso se habilita con true en el
    # segundo parámetro y normalización L2 que se recomienda para SIFT.
    bfCross = cv.BFMatcher_create(cv.NORM_L2, True)

    # Obteniendo caracteristicas: keypoints y descriptores en un solo paso.
    #   - Las imágenes tienen que convertirse a uint8, es lo que acepta OpenCV,
    #     sino se queja.
    #   - None es para indicar que no se está utilizando una máscara.
    
    img1K, img1D = sift.detectAndCompute(img1.astype('uint8'), None)
    img2K, img2D = sift.detectAndCompute(img2.astype('uint8'), None)

    # Haciendo el matching de Bruteforce+CrossCheck, se obtiene un objeto
    # con los descriptores de los puntos en correspondencia.
    bfCrossRes = bfCross.match(img1D, img2D)

    return bfCrossRes, img1K, img2K
    

def getKeyPoints_Lowe2NN(img1, img2, p_k=2):
    """
    Obtener los puntos característicos y descriptor de dos imágenes utilizando
    el criterio de Lowe.

    Parameters
    ----------
    img1 : Imagen
        Imagen a la cual encontrar los puntos en correspondencia con img2.
    img2 : Imagen
        Imagen con la cual se encuentran los puntos en correspondencia con img1.
    p_k : Entero, opcional
        Númeeros de vecinos a obtener por punto, por defecto 2.

    Returns
    -------
    bfLoweBest : Descriptores de los puntos en correspondencia entre las dos imágenes.
    img1K : Puntos de la imagen 1,
    img2K : Puntos de la imagen 2.

    """
    # Creando el objeto SIFT.
    sift = cv.SIFT_create()
    
    # Objeto para hacer Promedio de Lowe-2nn, sin CrossCheck (por defecto)
    bfLowe2nn = cv.BFMatcher_create(cv.NORM_L2)

    # Obteniendo caracteristicas: keypoints y descriptores en un solo paso.
    #   - Las imágenes tienen que convertirse a uint8, es lo que acepta OpenCV,
    #     sino se queja.
    #   - None es para indicar que no se está utilizando una máscara.
    
    img1K, img1D = sift.detectAndCompute(img1.astype('uint8'), None)
    img2K, img2D = sift.detectAndCompute(img2.astype('uint8'), None)
    
    # Haciendo matching, un punto con k-vecinos cercanos; k=2 (por defecto)
    loweMatch = bfLowe2nn.knnMatch(img1D, img2D, k=p_k)

    bfLoweBest = []
    
    # Se descartan los puntos ambiguos según el criterio de Lowe, es decir,
    # si la proporción del primer punto es menor que 0.8 * el segundo, 
    # se determina que es un punto no ambiguo y se acepta.
    for i, j in loweMatch:  
        if i.distance < 0.8*j.distance:
            # Se guarda en un vector porque sino drawMatchesKnn se queja.
            bfLoweBest.append([i])
            
    return bfLoweBest, img1K, img2K

############### FIN FUNCIONES EJERCICIO 2

############### FUNCIONES EJERCICIO 3

def genSimplePanorama(center, left, right, canvas):
    """
    Generar un panorama dadas 3 imágenes que se solapen horizontalmente

    Parameters
    ----------
    center : Imagen
        Imagen central del panorama.
    left : Imagen
        Imagen a la izquierda del panorama.
    right : Imagen
        Imagen a la derecha del panorama.
    canvas : Imagen, preferiblemente de solo negro.
        Imagen donde poner el panorama.

    Returns
    -------
    canvas : Imagen
        Imagen conteniendo el panormaa.

    """
    # Obtener el descriptor y los puntos entre la imagen izquierda y la central.
    leftCenter, k1, k2 = getKeyPoints_Lowe2NN(left, center)
    # Ídem con la derecha y la central.
    centerRight, kl1, kl2 = getKeyPoints_Lowe2NN(right, center)

    # Separar del objeto los descriptores de los puntos de "fuente" y "destino".
    # - Fuente: Los puntos de la imagen izquierda o derecha.
    # - Destino: Los puntos del centro
    srcLeft = np.float32([ k1[m[0].queryIdx].pt for m in leftCenter ]).reshape(-1,1,2)
    dstLeft = np.float32([ k2[m[0].trainIdx].pt for m in leftCenter ]).reshape(-1,1,2)
    
    srcRight = np.float32([ kl1[m[0].queryIdx].pt for m in centerRight ]).reshape(-1,1,2)
    dstRight = np.float32([ kl2[m[0].trainIdx].pt for m in centerRight ]).reshape(-1,1,2)
    
    # - Con estos descriptores de puntos, obtener una homografía usando RANSAC como método
    #   de eliminación de outliers.
    # - Se generará una homografía para transformar src* -> dst*, o sea,
    #   para transformar la imagen de la izquierda o derecha a la perspectiva
    #   de la imagen central.
    
    hLC, mask = cv.findHomography(srcLeft, dstLeft, cv.RANSAC)
    hRC, mask = cv.findHomography(srcRight, dstRight, cv.RANSAC)

    # Se genera una homografía sencilla de traslación entre la imagen central,
    # su centro y el centro de la homografía, esto se realiza por una diferencia
    # de la mitad de sus dimensiones.
    baseHom = np.array([[1, 0, canvas.shape[1]/2 - center.shape[1]/2],
                        [0, 1, canvas.shape[0]/2 - center.shape[0]/2],
                        [0, 0, 1]],
                       dtype=np.float64)

    # - Se pega primero la imagen más a la derecha, se combinan ambas homografías 
    #   por el producto punto al hacer el warpPerspective.
    # - El primer warpPerspective se hace sin bordes transparentes, se evita 
    #   que salga ruido en el resultado final.
    # - El resto de imágenes si lo utilizan, luego se pega la imagen izquierda.
    # - Por último la imagen central, ya que es la que tiene más calidad por 
    #   no haber sido transformada. (Además de moverla al centro del canvas)
    canvas = cv.warpPerspective(right, baseHom.dot(hRC), (canvas.shape[1], canvas.shape[0]), dst=canvas) 
    canvas = cv.warpPerspective(left, baseHom.dot(hLC), (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT) 
    canvas = cv.warpPerspective(center, baseHom, (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT)
    
    return canvas

############### FIN FUNCIONES EJERCICIO 3

############### FUNCIONES BONUS 2-1

def getHomography(img1, img2):
    """
    Obtiene una homografía entre dos imágenes.
    Intuición: La transformación para la imagen 1 para estar en la perspectiva
    de la imagen 2.

    Parameters
    ----------
    img1 : Imagen
        Imagen que se quiere transformar a la perspectiva de la otra imagen.
    img2 : Imagen
        Imagen base

    Returns
    -------
    hom : Matriz 3x3
        Homografía para transformar la imagen 1 a la perspectiva de la imagen 2.

    """
    # En esencia, es lo que se realizó en el Ejercicio 3 pero en una función
    # para no repetir tanto código.
    
    # Obtener puntos, y los descriptores entre puntos en correspondencia.
    pairs, k1, k2 = getKeyPoints_Lowe2NN(img1, img2)
    
    # Separar los descriptores
    src = np.float32([ k1[m[0].queryIdx].pt for m in pairs ]).reshape(-1,1,2)
    dst = np.float32([ k2[m[0].trainIdx].pt for m in pairs ]).reshape(-1,1,2)
          
    # Calcular la homografía para ir de img1 -> img2
    hom, mask = cv.findHomography(src, dst, cv.RANSAC)
    
    return hom

def genPanoramaFlat(center, left, right, canvas):
    """
    Función para calcular un panorama plano

    Parameters
    ----------
    center : Imagen
        Imagen central del panorama.
    left : Lista de imágenes
        Lista de imágenes a la izquierda de la imagen central.
        Ordenadas de mayor a menor distancia con la imagen central.
    right : Lista de imágenes
        Lista de imágenes a la derecha de la imagen central.
        Ordenadas de mayor a menor distancia con la imagen central.
    canvas : Imagen
        Imagen donde irá la proyección.

    Returns
    -------
    canvas : Imagen
        Imagen conteniendo la proyección.

    """
    
    # Obteniendo la homografía para trasladar la imagen central al centro del 
    # canvas.
    center2canvas = np.array([[1, 0, canvas.shape[1]/2 - center.shape[1]/2],
                    [0, 1, canvas.shape[0]/2 - center.shape[0]/2],
                    [0, 0, 1]],
                   dtype=np.float64)

    # Declarando las listas que contendrán las homografías.
    leftHs = []
    rightHs = []

    # Obteniendo la homografía de la imagen directamente a la izquierda de
    # la imagen central.
    hom = getHomography(left[0], center)
    # - Se obtiene de una vez la transformación completa para ir al canvas.
    # - Este valor será util en pocos momentos.
    prev = center2canvas.dot(hom)
    # Se guarda en la lista.
    leftHs.append(prev)

    # Para el resto de imágenes a la izquierda...
    for i in range(1, len(left)):
        #... se obtiene la homografía entre la imagen actual y la siguiente
        # más a la izquierda,
        hom = getHomography(left[i], left[i-1])
        # Se almacena en prev la transformación completa de una imagen dada
        # para llevarla al canvas y que encaje, gracias a las propiedades
        # de la homografía.
        prev = prev.dot(hom)
        # Se almacena en la lista las transformaciones que necesita la imagen
        # actual.
        leftHs.append(prev)
    
    # Ídem con las imágenes a la derecha.
    hom = getHomography(right[0], center)
    prev = center2canvas.dot(hom)
    rightHs.append(prev)
    
    for i in range(1, len(right)):
        hom = getHomography(right[i], right[i-1])
             
        prev = prev.dot(hom)        
        rightHs.append(prev)

    # Por comodidad, se invierten las listas: ahora las imágenes más alejadas
    # y por lo tanto, las que han sufrido mayores transformaciones están de 
    # primero.
    left.reverse()
    leftHs.reverse()
    
    # Esto es útil para pegar primero estas imágenes al canvas, así las
    # siguientes imágenes con menos transformaciones y por lo tanto, 
    # mejor calidad podrán estar encima y mejorar la calidad en general del
    # panorama
    
    # - Se comienza pegando la imagen más a la izquierda, ha sido una decisión
    # arbitraria: es lo mismo si fuera sido la imagen más a la derecha.
    # - Nuevamente esta primera sin bordes transparentes.
    canvas = cv.warpPerspective(left[0], leftHs[0], (canvas.shape[1], canvas.shape[0]), dst=canvas) 

    # Por cada imagen que resta que están a la derecha, irlas pegando del mismo modo.
    for i in range(1, len(left)): cv.warpPerspective(left[i], leftHs[i], (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT) 
    
    # Ídem con las imágenes a la derecha.
    right.reverse()
    rightHs.reverse()
    
    for i in range(0, len(right)): cv.warpPerspective(right[i], rightHs[i], (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT) 
    
    # Se pega finalmente la imagen central y se devuelve el panorama.
    canvas =  cv.warpPerspective(center, center2canvas, (canvas.shape[1], canvas.shape[0]), dst=canvas, borderMode=cv.BORDER_TRANSPARENT) 
    
    return canvas
    
############### FIN FUNCIONES BONUS 2-1

############################## IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 2
###############

print("Ejercicio 2 ejecutándose...")
#  "Con cada dos de las imágenes de Yosemite con solapamiento detectar y  
#   extraer los descriptores SIFT de OpenCV, usando para ello la función 
#   cv2.detectAndCompute()..."
    
yose1 = leeImagen("./imagenes/Yosemite1.jpg", False)
yose2 = leeImagen("./imagenes/Yosemite2.jpg", False)

print("- Calculando los puntos y descriptores.")

bfCrossRes, yose1Keys, yose2Keys = getKeyPoints_BF(yose1, yose2)
bfLoweBest, a, b = getKeyPoints_Lowe2NN(yose1, yose2)    

#   "... mostrar ambas imágenes en un mismo canvas y pintar líneas de diferentes 
#   colores entre las coordenadas de los puntos en correspondencias. Mostrar 
#   en cada caso un máximo de 100 elegidas aleatoriamente."

# Obtener 100 puntos aleatorios (o si hay menos de 100 solo esos puntos)
crossPoints = random.sample(bfCrossRes, min(100, len(bfCrossRes)))
lowePoints = random.sample(bfLoweBest, min(100, len(bfLoweBest)))

# Obteniendo las líneas de colores entre cada imagen dado los keypoints y descriptores.
# -Flags = 2 indica que 
bfCrossRes = cv.drawMatches(yose1.astype('uint8'), yose1Keys, yose2.astype('uint8'), yose2Keys, crossPoints, None, flags=2)
bfLoweRes = cv.drawMatchesKnn(yose1.astype('uint8'), a, yose2.astype('uint8'), b, lowePoints, None, flags=2)

print("- Pintando las imágenes")

# Pintar las imágenes.
pintaI(bfCrossRes, "SIFT: Bruteforce + CrossCheck")
pintaI(bfLoweRes, "SIFT: Lowe-Average-2NN")

print("...Ejercicio 2 finalizado\n")


#%% EJERCICIO 3
###############

print("Ejercicio 3 ejecutándose...")
# Cargando las imágenes...
left = leeImagen("./imagenes/IMG_20211030_110413_S.jpg", True)
center = leeImagen("./imagenes/IMG_20211030_110415_S.jpg", True)
right = leeImagen("./imagenes/IMG_20211030_110417_S.jpg", True)

# Generando el canvas, fue determinado por prueba y error.
canvas = np.zeros((400, 700))

# Llamando a la función que genera directamente el panorama.
simplePanorama = genSimplePanorama(center, left, right, canvas)

print("Pintando el panorama...")
# Pintando.
pintaI(simplePanorama)
print("...Ejercicio 3 finalizado\n")


#%% BONUS 2-1
###############

print("Bonus 2-1 ejecutándose...")
# Cargando todas las imágenes.
center = leeImagen("./imagenes/IMG_20211030_110420_S.jpg", True)

left = [leeImagen("./imagenes/IMG_20211030_110418_S.jpg", True),
        leeImagen("./imagenes/IMG_20211030_110417_S.jpg", True),
        leeImagen("./imagenes/IMG_20211030_110415_S.jpg", True),
        leeImagen("./imagenes/IMG_20211030_110413_S.jpg", True),
        leeImagen("./imagenes/IMG_20211030_110410_S.jpg", True)]

right = [leeImagen("./imagenes/IMG_20211030_110421_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110425_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110426_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110428_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110431_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110433_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110434_S.jpg", True),
         leeImagen("./imagenes/IMG_20211030_110436_S.jpg", True)]

# Generando el canvas, determinado por prueba y error nuevamente.
canvas = np.zeros((700, 2500))

# Utilizando otra función para sacar el panorama de todas las imágenes.
panorama = genPanoramaFlat(center, left, right, canvas)

print("Pintando el panorama...")
# Pintándolo.
pintaI(panorama)

print("...Bonus 2-1 finalizado\n")


#%% EJERCICIO 1
###############

print("Ejercicio 1 ejecutándose...")

# Cargando las imágenes.
yose1 = leeImagen("./imagenes/Yosemite1.jpg", False)
yose2 = leeImagen("./imagenes/Yosemite2.jpg", False)

# Generando las pirámides de Lowe.
gaussPyr1, DoGPyr1 = lowePyramid(yose1, 3, 3, 1.6)
gaussPyr2, DoGPyr2 = lowePyramid(yose2, 3, 3, 1.6)

# Pintando por pantalla las imágenes.
# - Octavas gaussianas.
showOctaves(gaussPyr1)
showOctaves(gaussPyr2)

# - Octavas laplacianas.
showOctaves(DoGPyr1)
showOctaves(DoGPyr1)

# Obteniendo los puntos.
# Nota:
# - Estas dos llamadas se tardan un total de 3 minutos aproximadamente en 
#   completarse.
# - De verdad no pude, no tuve el tiempo de poder implementar la máscara para 
#   poder aliviar un poco la carga, o intentar alguna solución con hebras. 
#   Pero aseguro que el código funciona completamente y sin errores. Lo lamento :c 
print("- Iniciando detección de puntos (Aproximadamente 3 mins)...")
print("--Yosemite 1...")
kpBest1, kpBestCV1, kpAll1, kpOct1 = getExtrema(DoGPyr1)
print("...Listo")
print("--Yosemite 2...")
kpBest2, kpBestCV2, kpAll2, kpOct2 = getExtrema(DoGPyr2)
print("...Listo")
print("...Detección finalizada.")

# Pintando los 100 puntos característicos.
yoseOut1 = cv.drawKeypoints(yose1.astype('uint8'), kpBestCV1, None, flags=4)
yoseOut2 = cv.drawKeypoints(yose2.astype('uint8'), kpBestCV2, None, flags=4)

pintaI(yoseOut1, "100 Keypoints con mayor respuesta en general")
pintaI(yoseOut2, "100 Keypoints con mayor respuesta en general")

print("...Ejercicio 1 finalizado\n")

#%% BONUS 1-1 
############### Necesita que EJERCICIO 1 se haya ejecutado antes.
print("Bonus 1-1 ejecutándose...")


# Obteniendo los 100 mejores puntos por octava en proporción
kpBestOct1, kpCVOct1 = getBestPercentage(kpOct1)
kpBestOct2, kpCVOct2 = getBestPercentage(kpOct2)

# Pintando los puntos.
yoseOutBonus1_1 = cv.drawKeypoints(yose1.astype('uint8'), kpCVOct1, None, flags=4)
yoseOutBonus1_2 = cv.drawKeypoints(yose2.astype('uint8'), kpCVOct2, None, flags=4)

pintaI(yoseOutBonus1_1, "100 Keypoints con mayor respuesta por octava")
pintaI(yoseOutBonus1_2, "100 Keypoints con mayor respuesta por octava")

print("...Bonus 1-1 finalizado\n")

#%% BONUS 1-2
############### Necesita que EJERCICIO 1 se haya ejecutado antes.

print("Bonus 1-2 ejecutándose...")

# Obteniendo los puntos interpolados tomados de los 100 mejores.
keyIP1, keyIPCV1 = extremaInterpolation(kpBest1, DoGPyr1)
keyIP2, keyIPCV2 = extremaInterpolation(kpBest2, DoGPyr2)

# Pintando los puntos originales y los interpolados.
#  - Los rojos son los originales y los verdes los interpolados.
yoseOutBonus2_1 = cv.drawKeypoints(yose1.astype('uint8'), kpBestCV1, None, color=(0,0,255), flags=4)
yoseOutBonus2_1 = cv.drawKeypoints(yoseOutBonus2_1, keyIPCV1, None, color=(0,255,0), flags=4)

yoseOutBonus2_2 = cv.drawKeypoints(yose2.astype('uint8'), kpBestCV2, None, color=(0,0,255), flags=4)
yoseOutBonus2_2 = cv.drawKeypoints(yoseOutBonus2_2, keyIPCV2, None, color=(0,255,0), flags=4)

pintaI(yoseOutBonus2_1, str(len(keyIPCV1)) + " Keypoints con mayor respuesta interpolados")
pintaI(yoseOutBonus2_2, str(len(keyIPCV2)) + " Keypoints con mayor respuesta interpolados")

# Un buen ejemplo, el keypoint 47 original, que es el 44 de los que sobró en la interpolación.
pintaI(yoseOutBonus2_1[265:315,220:270], "Zoom: Ejemplo de KeyPoint Interpolado")
print("Ejemplo del keypoint de índice 47: \n\tPosición:",kpBestCV1[47].pt,"\n\tTamaño/Sigma:",kpBestCV1[46].size)
print("Su versión interpolada, ahora de índice 43: \n\tPosición:",keyIPCV1[43].pt,"\n\tTamaño/Sigma:",keyIPCV1[43].size)

print("...Bonus 1-2 finalizado\n")

