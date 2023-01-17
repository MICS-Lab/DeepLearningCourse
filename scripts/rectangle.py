import keras
from keras.models import Sequential
from keras.layers import Dense
import os
from ann_visualizer.visualize import ann_viz

file_path = os.path.realpath(__file__)
os.chdir(file_path.replace("rectangle.py", ''))

rectangle_network = Sequential()
rectangle_network.add(Dense(units=5,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=1))
for i in range(3):
        rectangle_network.add(Dense(units=5,
                               activation='relu',
                               kernel_initializer='uniform'))
rectangle_network.add(Dense(units=1,
                  activation='sigmoid',
                  kernel_initializer='uniform'))
rectangle_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ann_viz(rectangle_network, title="wide", filename="rectangle.gv")
