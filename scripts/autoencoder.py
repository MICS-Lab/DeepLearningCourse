import keras
from keras.models import Sequential
from keras.layers import Dense
import os
from ann_visualizer.visualize import ann_viz

file_path = os.path.realpath(__file__)
os.chdir(file_path.replace("autoencoder.py", ''))

ae_network = Sequential()
ae_network.add(Dense(units=5,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=9))
ae_network.add(Dense(units=3,
                activation='sigmoid',
                kernel_initializer='uniform'))
ae_network.add(Dense(units=5,
                activation='relu',
                kernel_initializer='uniform'))
ae_network.add(Dense(units=9,
                  activation='sigmoid',
                  kernel_initializer='uniform'))
ae_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ann_viz(ae_network, title="autoencoder", filename="autoencoder.gv")
