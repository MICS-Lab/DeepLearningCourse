import keras
from keras.models import Sequential
from keras.layers import Dense
import os
from ann_visualizer.visualize import ann_viz

file_path = os.path.realpath(__file__)
os.chdir(file_path.replace("gan.py", ''))

gan_network = Sequential()
gan_network.add(Dense(units=9,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=3))
gan_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ann_viz(gan_network, title="gan", filename="gan.gv")
