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

import tensorflow.keras.utils as np_utils

from keras.models import Sequential

# Importar el conjunto de datos
from keras.datasets import cifar100

# Imports para hacer el conjunto de validación
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from keras.preprocessing.image import load_img,img_to_array


import time

# Importando nombres de las capas por legibilidad

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout

# -- Ejercicio 3 --
# Cosas para ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
  y_train_v = np_utils.to_categorical(y_train, 25)
  y_test_v = np_utils.to_categorical(y_test, 25)
  
  return x_train, y_train_v, x_test, y_test_v


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
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)
  
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
        
        
    
def getBaseNet():
    #########################################################################
    ################## DEFINICIÓN DEL MODELO BASENET ########################
    #########################################################################

    # Definir model e incluir las capas en él
    model = Sequential(name="initialModel")
    
    #   1 - Conv2D
    #   2 - ReLU
    #           - 6 canales de salida
    #           - Tamaño de kernel 5
    #           - Entrada es una imagen 32x32x3 (RGB), se tiene que indicar
    #           cuando es la primera capa de la red según la documentación.
    #           - La salida tendrá dimensión 28 ya que el filtro no tiene padding.
    model.add(Conv2D(6, 5, activation='relu', input_shape = (32, 32, 3)))
    
    #   3 - MaxPooling2D
    #           - Kernel de 2x2, reduce la imagen a la mitad, de 28x28 a 14x14
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #   4 - Conv2D
    #   5 - ReLU
    #           - 16 canales de salida
    #           - Tamaño de kernel 5
    #           - La salida tendrá dimensión 10 ya que el filtro no tiene padding.
    model.add(Conv2D(16, 5, activation="relu"))
    
    #   6 - MaxPooling2D
    #           - Kernel de 2x2, reduce la imagen a la mitad, de 10x10 a 5x5
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #   Se aplasta con flatten, para conectarlo con las capas totalmente conectadas.
    model.add(Flatten())
    
    #   7 - Linear / Dense
    #   9 - ReLU
    #           - La dimensión de salida será de 50 unidades, que son 50 neuronas.
    model.add(Dense(50, activation='relu'))
    
    #   9 - Linear / Dense
    #           - Se tienen 25 neuronas, o sea que la salida será un vector de 25
    #           unidades.
    model.add(Dense(25, activation='softmax'))
    
    return model


def compileModel(model):
    
    model.compile(loss = k.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])
    

def threeFoldCrossVal(model, batchSize, numEpoch, trainImg, trainTag, testImg, testTag):
    # Merge inputs and targets
    inputs = np.concatenate((trainImg, testImg), axis=0)
    targets = np.concatenate((trainTag, testTag), axis=0)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=3, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
      # Generate a print
      print('------------------------------------------------------------------------')
      print(f'Training for fold {fold_no} ...')
    
      # Fit data to model
      history = model.fit(inputs[train],
                  batch_size=batchSize,
                  epochs=numEpoch,
                  verbose=1)
    
      # Generate generalization metrics
      scores = model.evaluate(inputs[test], targets[test], verbose=0)
      print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
      # acc_per_fold.append(scores[1] * 100)
      # loss_per_fold.append(scores[0])
    
      # Increase fold number
      fold_no = fold_no + 1

############### FIN FUNCIONES EJERCICIO 1

############### FUNCIONES EJERCICIO 2
def getNormalizedData(trainImg, trainTag, batchSize, testImg):
    dataG = ImageDataGenerator(featurewise_center = True,
                                featurewise_std_normalization = True,
                                 rotation_range=10, # rotation
                                 width_shift_range=0.2, # horizontal shift
                                 height_shift_range=0.2, # vertical shift
                                 horizontal_flip=True)
    dataG.fit(trainImg)
    
    X_train, X_val, Y_train, Y_val = train_test_split(trainImg, trainTag,
                                  test_size=0.1, stratify=trainTag)
    
    it_train = dataG.flow(X_train , Y_train, batch_size = batchSize)
    it_validation = dataG.flow(X_val , Y_val)


    dataT = ImageDataGenerator(featurewise_center = True,
                               featurewise_std_normalization = True)
    
    dataT.fit(trainImg)
    dataT.standardize(testImg)

    return it_train, it_validation, testImg


def getBaseNetPlusPlus():
    #########################################################################
    ########################## MEJORA DEL MODELO ############################
    #########################################################################
    
    # Definir model e incluir las capas en él
    model = Sequential(name="augmentedModel")
    
    # 3º Cambio 6 -> 256
    model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape = (32, 32, 3)))    
    model.add(BatchNormalization())                 # 1º Cambio
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())                 # 1º Cambio
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 3º Cambio 16 -> 512
    model.add(Conv2D(64, 5, activation="relu"))
    model.add(BatchNormalization())                 # 1º Cambio

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    # 3º Cambio 50 -> 128
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    # 4º Cambio

    model.add(Dense(25, activation='softmax'))

    
    # # A completar. Tanto la normalización de los datos como el data
    # # augmentation debe hacerse con la clase ImageDataGenerator.
    # # Se recomienda ir entrenando con cada paso para comprobar
    # # en qué grado mejora cada uno de ellos.
    
    # augmentedModel = Sequential(name="augmentedModel")
    
    # augmentedModel.add(Conv2D(32, 5, activation='relu', input_shape = (32, 32, 3)))
    # # 
    
    # augmentedModel.add(MaxPooling2D(pool_size=(2,2)))
    
    # augmentedModel.add(Conv2D(16, 5, activation="relu"))
    
    # augmentedModel.add(MaxPooling2D(pool_size=(2,2)))
    
    # augmentedModel.add(Flatten())
    # augmentedModel.add(Dense(50, activation='relu'))
    # # augmentedModel.add(Dropout(0.5))
    
    
    # # augmentedModel.add(Dense(128, activation='relu'))
    # # augmentedModel.add(Dropout(0.25))
    
    # augmentedModel.add(Dense(25, activation='softmax'))

    return model


############### FIN FUNCIONES EJERCICIO 2

############################## IMPLEMENTACION EJERCICIOS

#%% CUDA

# Utilizando CUDA si es posible
# - Se configura para que no use toda la memoria de una vez, sino que vaya 
# añadiendo lo que necesita en el entrenamiento.
# - Evita errores al momento de ejecutar.

configCUDA()

#%% EJERCICIO 1
###############

# Obteniendo las imágenes dada la función proporcionada
trainImg1, trainTag1, testImg1, testTag1 = cargarImagenes()

# Generando el modelo
baseNet = getBaseNet()

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

compileModel(baseNet)

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################


#%%


datagen = ImageDataGenerator(featurewise_center = True,
                          featurewise_std_normalization = True,
                          validation_split = 0.1)
    
datagen.fit(trainImg1)

testGen = ImageDataGenerator(featurewise_center = True,
                          featurewise_std_normalization = True)
testGen.fit(trainImg1)
testGen.standardize(testImg1)

# TAMAÑO DE BATCH
batchSize = 32

# ÉPOCAS DE ENTRENAMIENTO
trainEpochs = 100

# Incluir model.fit()
# hist = baseNet.fit(trainImg1, trainTag1, batch_size=batchSize,
#                  epochs=trainEpochs, validation_split=0.1,
#                  verbose=1)

hist = baseNet.fit(datagen.flow(trainImg1, 
                                trainTag1,
                                batch_size = 32,
                                subset= 'training'),
                   epochs = trainEpochs,
                   validation_data = datagen.flow(trainImg1, 
                                                  trainTag1,
                                                  batch_size = 32,
                                                  subset = 'validation'),
                   verbose = 1)
                    

# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(hist)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#Incluir model.evaluate() 
score = baseNet.evaluate(testImg1, testTag1, verbose=0)
#Incluir función que muestre la perdida y accuracy del test
print("Resultados de Evaluación en Test\n - Loss:", score[0])
probTag = baseNet.predict(testImg1) 
print(' - Accuracy: {}%'.format(calcularAccuracy(probTag, testTag1)*100)) #Este valor coincide con score[1]


# threeFoldCrossVal(model, batchSize, trainEpochs, trainImg1, train, testImg1, test)


#%% EJERCICIO 2
###############

# Obteniendo las imágenes dada la función proporcionada
trainImg2, trainTag2, testImg2, testTag2 = cargarImagenes()


augmentedModel = getBaseNetPlusPlus()

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

#%
# incluir model.compile()
augmentedModel.compile(loss = k.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])


#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

#%

# TAMAÑO DE BATCH
batchSize = 32

# ÉPOCAS DE ENTRENAMIENTO
trainEpochs = 100

trainIter, valIter, normTest = getNormalizedData(trainImg2, trainTag2, batchSize, testImg2)


# datagen.fit(trainImg1)

# testGen = ImageDataGenerator(featurewise_center = True,
#                           featurewise_std_normalization = True)
# testGen.fit(trainImg1)
# testGen.standardize(testImg1)

# # TAMAÑO DE BATCH
# batchSize = 32

# # ÉPOCAS DE ENTRENAMIENTO
# trainEpochs = 100

# # Incluir model.fit()
# # hist = baseNet.fit(trainImg1, trainTag1, batch_size=batchSize,
# #                  epochs=trainEpochs, validation_split=0.1,
# #                  verbose=1)

# hist = baseNet.fit(datagen.flow(trainImg1, 
#                                 trainTag1,
#                                 batch_size = 32,
#                                 subset= 'training'),
#                    epochs = trainEpochs,
#                    validation_data = datagen.flow(trainImg1, 
#                                                   trainTag1,
#                                                   batch_size = 32,
#                                                   subset = 'validation'),
#                    verbose = 0)

#%%
# Incluir model.fit()
hist = augmentedModel.fit(trainIter, batch_size=batchSize,
                 epochs=trainEpochs, validation_data = valIter, verbose=1)


# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(hist)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#Incluir model.evaluate() 
score = augmentedModel.evaluate(testImg2, testTag2, verbose=0)
#Incluir función que muestre la perdida y accuracy del test
print("Test loss =", score[0])
print("Test accuracy=", score[1])

probTag2 = augmentedModel.predict(testImg2) 
print('Accuracy de nuestro modelo en test: {}%'.format(calcularAccuracy(probTag2, testTag2)*100)) #Este valor coincide con score[1]



#%% EJERCICIO 3
###############

"""## Usar ResNet50 preentrenada en ImageNet como un extractor de características"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.

trainGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)
testGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)

trainImg3, trainClass3, testImg3, testClass3 = cargarDatos("./imagenes")

#%%
# Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
# A completar
resNet  = ResNet50(include_top = False, 
                   weights = 'imagenet',
                   pooling = 'avg',
                   input_shape= (224, 224, 3))

# Extraer las características las imágenes con el modelo anterior.

trainFeatures = resNet.predict(trainGenerator.flow(trainImg3, batch_size = 1, shuffle=False), 
                                                   verbose = 1)

testFeatures = resNet.predict(testGenerator.flow(testImg3, batch_size= 1, shuffle=False), 
                                                 verbose = 1)

# Las características extraídas en el paso anterior van a ser la entrada
# de un pequeño modelo de dos capas Fully Conected, donde la última será la que 
# nos clasifique las clases de Caltech-UCSD (200 clases). De esta forma, es 
# como si hubiéramos fijado todos los parámetros de ResNet50 y estuviésemos
# entrenando únicamente las capas añadidas. Definir dicho modelo.
# A completar: definición del modelo, del optimizador y compilación y
# entrenamiento del modelo.

#%%
miniModel = Sequential()

miniModel.add(Dense(1024, activation='relu', input_shape = (2048,)))
miniModel.add(Dense(512, activation='relu'))
miniModel.add(Dropout(0.5))
miniModel.add(Dense(200, activation='softmax'))

miniModel.compile(loss = k.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])


# En la función fit() puedes usar el argumento validation_split

hist = miniModel.fit(trainFeatures, trainClass3, batch_size = 32, 
                     epochs = 100, validation_split=0.1)


# Incluir función que muestre la evolución del entrenamiento y validación
mostrarEvolucion(hist)      

probTag3 = miniModel.predict(testFeatures) 
print('Accuracy de nuestro modelo en test: {}%'.format(calcularAccuracy(probTag3, testClass3)*100)) #Este valor coincide con score[1]


"""## Reentrenar ResNet50 (fine tunning)"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar


# Añadir nuevas capas al final de ResNet50 (recuerda que es una instancia de
# la clase Model).


# Compilación y entrenamiento del modelo.
# A completar.