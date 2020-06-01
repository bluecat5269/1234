# Import some useful packages
import matplotlib.pyplot as plt
import numpy as np

# Layers for FNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# Layers for CNN
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import SGD, Adam

# For data preprocessing
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

#Functional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate, add

f_1 = Dense(500, activation='sigmoid')
f_2 = Dense(500, activation='sigmoid')
f_3 = Dense(256, activation='sigmoid')
f_4 = Dense(256, activation='sigmoid')
f_5 = Dense(10, activation='softmax')

x = Input(shape=(784,))
print(x)

y=Input(shape=(392,))
print(y)

h_1 = f_1(x)
h_2 = f_2(y)
h_3 = f_3(h_1)
h_4 = f_4(h_2)
u = concatenate([h_3,h_4])#合併
z=f_5(u)

model = Model([x,y], z)
model.summary()