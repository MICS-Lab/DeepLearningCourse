import keras
from keras.models import Sequential
from keras.layers import Dense
import os
from ann_visualizer.visualize import ann_viz

file_path = os.path.realpath(__file__)
os.chdir(file_path.replace("wide_vs_deep.py", ''))

wide_network = Sequential()
wide_network.add(Dense(units=10,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=1))
wide_network.add(Dense(units=1,
                  activation='sigmoid',
                  kernel_initializer='uniform'))
wide_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


deep_network = Sequential()
deep_network.add(Dense(units=2,
                        activation='relu',
                        kernel_initializer='uniform',
                        input_dim=1))
for i in range(4):
        deep_network.add(Dense(units=2,
                               activation='relu',
                               kernel_initializer='uniform'))
deep_network.add(Dense(units=1,
                        activation='sigmoid',
                        kernel_initializer='uniform'))
deep_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


ann_viz(wide_network, title="wide", filename="wide.gv")
ann_viz(deep_network, title="deep", filename="deep.gv")
