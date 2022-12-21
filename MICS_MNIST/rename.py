import os
import glob
import shutil

print(os.getcwd())

for k in range(10):
    splitted_names =  glob.glob(f'MICS_MNIST/splitted_clean/{k}_*.png')
    print(k, len(splitted_names))
    for i, splitted_name in enumerate(splitted_names):
        # fsting with leading zeros
        # https://stackoverflow.com/questions/339007/nicest-way-to-pad-zeroes-to-string
        shutil.copy(splitted_name, f'MICS_MNIST/splitted_final/{k}_{i:02}.png')
