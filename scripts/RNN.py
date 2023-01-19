import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input
import os
from ann_visualizer.visualize import ann_viz

file_path = os.path.realpath(__file__)
os.chdir(file_path.replace("RNN.py", ''))


network = Sequential()
network.add(Dense(units=8,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=10))

ann_viz(network, title="RNN", filename="rnn.gv")
