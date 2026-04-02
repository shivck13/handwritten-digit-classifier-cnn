# print python, tf, keras, opencv versions and os version
import sys
import os
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import matplotlib
import numpy as np

def print_versions():
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("OpenCV version:", cv2.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Numpy version:", np.__version__)
    print("OS version:", os.name)

if __name__ == "__main__":
    print_versions()