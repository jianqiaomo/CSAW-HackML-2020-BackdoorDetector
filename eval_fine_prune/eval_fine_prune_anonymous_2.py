import fine_prune
import sys
import os
import sys
import random
import h5py
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import keras
# from keras.utils.generic_utils import CustomObjectScope
from matplotlib import pyplot as plt

class G(keras.Model):
  def __init__(self, B, B_prime):
      super(G, self).__init__()
      self.B = B
      self.B_prime = B_prime

  def predict(self,data):
      y = np.argmax(self.B.predict(data), axis=1)
      y_prime = np.argmax(self.B_prime.predict(data), axis=1)
      pred = 0
      if y==y_prime:
          pred = y
      else:
          pred = 1283
      return pred

if __name__ == '__main__':
    
    model_path = '../models_G/anonymous2_prune_net.h5'
    img_path = sys.argv[1]
    
    model_filename = str('../models/anonymous_2_bd_net.h5')
    bd_model = keras.models.load_model(model_filename)

    b_prime_model = load_model(model_path)
    G_model_X = G(bd_model, b_prime_model)
    x = plt.imread(img_path)[:,:,0:3]


    pred = G_model_X.predict(x[None])
    print(pred)








