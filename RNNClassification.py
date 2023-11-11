import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2

Img_Size = 224
Batch_Size = 64
epochs = 10

max_seq_length = 20
num_features = 2048

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


