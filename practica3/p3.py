#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 3: Redes Neuronales Convolucionales
    Asignatura: Vision por Computador
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Diciembre 2021
    
[ENGLISH]

    Practice 3: Convolutional Neural Networks
    Course: Computer Visionhuh
    Author: Valentino Lugli (Github: @RhinoBlindado)
    December 2021

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

#   Usando keras y tensorflow para las redes
import tensorflow as tf
import tensorflow.keras as k

import keras.utils as np_utils
from keras.models import Sequential

# Importar el conjunto de datos
from keras.datasets import cifar100

# Imports para hacer el conjunto de validación
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import time


# FUNCIONES AUXILIARES

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función sólo se le llama una vez. Devuelve 4 vectores conteniendo,
# por este orden, las imágenes de entrenamiento, las clases de las imágenes
# de entrenamiento, las imágenes del conjunto de test y las clases del
# conjunto de test.

def cargarImagenes():
  # Cargamos Cifar100. Cada imagen tiene tamaño (32, 32, 3).
  # Nos vamos a quedar con las imágenes de 25 de las clases.
  
  (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  
  train_idx = np.isin(y_train, np.arange(25))
  train_idx = np.reshape(train_idx,-1)
  x_train = x_train[train_idx]
  y_train = y_train[train_idx]
  
  test_idx = np.isin(y_test, np.arange(25))
  test_idx = np.reshape(test_idx, -1)
  x_test = x_test[test_idx]
  y_test = y_test[test_idx]
  
  # Transformamos los vectores de clases en matrices. Cada componente se convierte en un vector
  # de ceros con un uno en la componente correspondiente a la clase a la que pertenece la imagen.
  # Este paso es necesario para la clasificación multiclase en keras.
  y_train = np_utils.to_categorical(y_train, 25)
  y_test = np_utils.to_categorical(y_test, 25)
  
  return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()

############### FIN FUNCIONES AUXILIARES

############################## FUNCIONES PARA EJERCICIOS 

############### FUNCIONES EJERCICIO 1
def configCUDA():
    # Usando CUDA si es posible.

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
        
def threeFoldCrossVal():
    pass

############### FIN FUNCIONES EJERCICIO 1

############### FUNCIONES EJERCICIO 2
# def getNormalizedData(trainImg, trainTag):
#     dataG = ImageDataGenerator(featurewise_center = True,
#                                featurewise_std_normalization = True)
#     dataG.fit(trainImg)
#     dataG = ImageDataGenerator(validation_split = 0.1)
    
#     X_train, X_val, Y_train, Y_val = train_test_split(trainImg, trainTag,
#                                   test_size=0.1, stratify=trainTag)
    
#     it_train = datagen.dataG(X_train , Y_train, batch_size = batch_size)
#     it_validation = datagen.dataG(X_val , Y_val)



############### FIN FUNCIONES EJERCICIO 2


############################## IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1
###############

# - Utilizando CUDA si es posible, se configura para que no use toda la memoria
# de una vez, sino que vaya añadiendo lo que necesita en el entrenamiento.
# - Evita errores al momento de ejecutar.
# - Si se usa CPU, se ignora la función.
configCUDA()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# A completar
# 1.- Incluir import del tipo de modelo y capas a usar
#   Ya incluido en las cabeceras

# 2.- definir model e incluir las capas en él
model = Sequential(name="initialModel")

#   2.1 - Conv2D:
#           - 6 canales de salida
#           - Tamaño de kernel 5
#           - Usando ReLU (capa 2.2)
#           - Entrada es una imagen 32x32x3 (RGB), se tiene que indicar
#           cuando es la primera capa de la red según la documentación.
#           - La salida tendrá dimensión 28 ya que el filtro no tiene padding.
model.add(k.layers.Conv2D(6, 5, activation='relu', input_shape = (32, 32, 3)))

#   2.3 - MaxPooling2D
#           - Kernel de 2x2, reduce la imagen a la mitad, de 28x28 a 14x14
model.add(k.layers.MaxPooling2D(pool_size=(2,2)))

#   2.4 - Conv2D
#           - 16 canales de salida
#           - Tamaño de kernel 5
#           - Usando ReLU (capa 2.5)
#           - La salida tendrá dimensión 10 ya que el filtro no tiene padding.
model.add(k.layers.Conv2D(16, 5, activation="relu"))

#   2.6 - MaxPooling2D
#           - Kernel de 2x2, reduce la imagen a la mitad, de 10x10 a 5x5
model.add(k.layers.MaxPooling2D(pool_size=(2,2)))

#   Se aplasta con flatten, para conectarlo con las capas totalmente conectadas.
model.add(k.layers.Flatten())

#   2.7 - Linear / Dense
#           - La dimensión de salida será de 50 unidades, que son 50 neuronas.
#           - Usando ReLU (capa 2.8)
model.add(k.layers.Dense(50, activation='relu'))

#   2.8 - Linear / Dense
#           - Se tienen 25 neuronas, o sea que la salida será un vector de 25
#           unidades.
model.add(k.layers.Dense(25, activation='softmax'))

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

#%
# incluir model.compile()
model.compile(loss = k.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Obteniendo las imágenes dada la función proporcionada
trainImg, trainTag, testImg, testTag = cargarImagenes()


#%%

# TAMAÑO DE BATCH
batchSize = 32

# ÉPOCAS DE ENTRENAMIENTO
trainEpochs = 50

# Incluir model.fit()
hist = model.fit(trainImg, trainTag, batch_size=batchSize,
                 epochs=trainEpochs, validation_split=0.1,
                 verbose=1)
# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(hist)


#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#Incluir model.evaluate() 
score = model.evaluate(testImg, testTag, verbose=0)
#Incluir función que muestre la perdida y accuracy del test
print("Test loss =", score[0])
print("Test accuracy=", score[1])

probTag = model.predict(testImg) 
print('Accuracy de nuestro modelo en test: {}%'.format(calcularAccuracy(probTag, testTag)*100)) #Este valor coincide con score[1]



#%% EJERCICIO 2
############### Depende de Ejercicio 1 por los pesos iniciales.

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

augmentedModel = Sequential(name="augmentedModel")

#   2.1 - Conv2D:
augmentedModel.add(k.layers.Conv2D(128, 3, activation='relu', input_shape = (32, 32, 3)))
augmentedModel.add(k.layers.Conv2D(128, 3, activation='relu'))


#   2.3 - MaxPooling2D
augmentedModel.add(k.layers.MaxPooling2D(pool_size=(2,2)))

#   2.4 - Conv2D
augmentedModel.add(k.layers.Conv2D(128, 3, activation="relu"))
augmentedModel.add(k.layers.Conv2D(64, 3, activation='relu'))
# augmentedModel.add(k.layers.Conv2D(128, 3, activation='relu'))


#   2.6 - MaxPooling2D
augmentedModel.add(k.layers.MaxPooling2D(pool_size=(2,2)))
augmentedModel.add(k.layers.Flatten())

#   2.7 - Linear / Dense
augmentedModel.add(k.layers.Dense(512, activation='relu'))
augmentedModel.add(k.layers.Dropout(0.5))


augmentedModel.add(k.layers.Dense(256, activation='relu'))
augmentedModel.add(k.layers.Dropout(0.5))

augmentedModel.add(k.layers.Dense(128, activation='relu'))
augmentedModel.add(k.layers.Dropout(0.5))

#   2.8 - Linear / Dense
augmentedModel.add(k.layers.Dense(25, activation='softmax'))


#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

#%
# incluir model.compile()
augmentedModel.compile(loss = k.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])

# Utilizando los mismos pesos que en el modelo base.

# augmentedModel.set_weights(weights)

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

#%



datagen = k.preprocessing.image.ImageDataGenerator(featurewise_center = True, 
                                                   featurewise_std_normalization = True)
datagen.fit(trainImg)


# TAMAÑO DE BATCH
batchSize = 32

# ÉPOCAS DE ENTRENAMIENTO
trainEpochs = 50


#%
# Incluir model.fit()
hist = augmentedModel.fit(trainImg, trainTag, batch_size=batchSize,
                 epochs=trainEpochs, validation_split=0.1, verbose=1)


# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(hist)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#Incluir model.evaluate() 
score = augmentedModel.evaluate(testImg, testTag, verbose=0)
#Incluir función que muestre la perdida y accuracy del test
print("Test loss =", score[0])
print("Test accuracy=", score[1])

probTag = augmentedModel.predict(testImg) 
print('Accuracy de nuestro modelo en test: {}%'.format(calcularAccuracy(probTag, testTag)*100)) #Este valor coincide con score[1]



#%% EJERCICIO 3
###############

#########################################################################
################### OBTENER LA BASE DE DATOS ############################
#########################################################################

# Descargar las imágenes de http://www.vision.caltech.edu/visipedia/CUB-200.html
# Descomprimir el fichero.
# Descargar también el fichero list.tar.gz, descomprimirlo y guardar los ficheros
# test.txt y train.txt dentro de la carpeta de imágenes anterior. Estos 
# dos ficheros contienen la partición en train y test del conjunto de datos.

#########################################################################
################ CARGAR LAS LIBRERÍAS NECESARIAS ########################
#########################################################################

# Terminar de rellenar este bloque con lo que vaya haciendo falta

# Importar librerías necesarias
import numpy as np
import keras
import keras.utils as np_utils
from keras.preprocessing.image import load_img,img_to_array
import matplotlib as plt

# Importar el optimizador a usar
from keras.optimizers import SGD

# Importar modelos y capas específicas que se van a usar


# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesario pasarle a las imágenes para usar este modelo


# Importar el optimizador a usar
from keras.optimizers import SGD

#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y 
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img, 
                                             target_size = (224, 224))) 
                       for img in vec_imagenes])
  return imagenes, clases

#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las 
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)
  
  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images)
  test, test_clases = leerImagenes(test_images)
  
  # Pasamos los vectores de las clases a matrices 
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = np_utils.to_categorical(train_clases, 200)
  test_clases = np_utils.to_categorical(test_clases, 200)
  
  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]
  
  return train, train_clases, test, test_clases

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()

  acc = hist.history['acc']
  val_acc = hist.history['val_acc']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()

"""## Usar ResNet50 preentrenada en ImageNet como un extractor de características"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar


# Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
# A completar
restNet  = tf.keras.applications.resnet50.ResNet50(include_top= False)

# Extraer las características las imágenes con el modelo anterior.
# A completar

# Las características extraídas en el paso anterior van a ser la entrada
# de un pequeño modelo de dos capas Fully Conected, donde la última será la que 
# nos clasifique las clases de Caltech-UCSD (200 clases). De esta forma, es 
# como si hubiéramos fijado todos los parámetros de ResNet50 y estuviésemos
# entrenando únicamente las capas añadidas. Definir dicho modelo.
# A completar: definición del modelo, del optimizador y compilación y
# entrenamiento del modelo.
# En la función fit() puedes usar el argumento validation_split

"""## Reentrenar ResNet50 (fine tunning)"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar


# Añadir nuevas capas al final de ResNet50 (recuerda que es una instancia de
# la clase Model).


# Compilación y entrenamiento del modelo.
# A completar.