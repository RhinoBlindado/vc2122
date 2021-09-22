# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:22:16 2013

@author: NPB
"""

print (__doc__)

#import: realiza en python una función equivalente a los includes en C/C++
#cualquier función que se desee usar ha debido ser importada previamente
#ya sea de forma directa o a través del modulo  en el que este o del paquete 
#que contiene al módulo

import random #generador de numeros aleatorios
#import numpy as np  # importa el paquete de  funciones de cálculo 
import pylab as pl  # importa el paquete de fucniones de gráficos

#  importamos dos modulos del paquete sklearn
from sklearn import svm, datasets
#  importamos uan función del modulo sklearn.metrics
from sklearn.metrics import confusion_matrix

#leemos un fichero  de datos usando funciones definidas en datasets
iris = datasets.load_iris()
# iris es un objeto que contiene entre otros dos metodos relevantes 
X= iris.data
y=iris.target
# estaremos filas y columnas de X
n_samples, n_features = X.shape
#generamos un vector con valores de 1....n_samples
p = range(n_samples)
#fijamos una semilla al generador de numeros aleatorios
random.seed(0)
#realizamos una permutación aleatorio del vector p
random.shuffle(p)
#asignamos los valores de X e Y en el orden de p
X, y = X[p], y[p]
#calculamos la mitad entera de n_samples
half = int(n_samples / 2)

# Run classifier
# definimos el calsificador a usar y los parametros
classifier = svm.SVC(kernel='linear')
#ajustamos el clasificador con los datos de entrenamiento
y_ = classifier.fit(X[:half], y[:half]).predict(X[half:])

# Compute confusion matrix. La evaluación de los resultados
cm = confusion_matrix(y[half:], y_)

#escribimos la matriz
print (cm)

# Show confusion matrix
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()