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

def get_prune_layer(layer):
  if layer.name in ['conv_3']:
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.8, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}


def data_loader(filepath):
      data = h5py.File(filepath, 'r')
      x_data = np.array(data['data'])
      y_data = np.array(data['label'])
      x_data = x_data.transpose((0,2,3,1))
      return x_data, y_data

def data_preprocess(x_data):
       return x_data/255.0

if __name__ == '__main__':
    
    model_path = '../models_G/sunglasses_prune_net.h5'
    img_path = sys.argv[1]
    
    model_filename = str('../models/anonymous_1_bd_net.h5')
    bd_model = keras.models.load_model(model_filename)
    # bd_model.compile(optimizer='adam',
    #                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                        metrics=['accuracy'])
   

    
    # layer = bd_model.get_layer('conv_3')
    # prune_layer = get_prune_layer(layer)
    # b_prime_model = load_model(model_path,custom_objects={'PruneLowMagnitude':prune_layer})

    b_prime_model = load_model(model_path)
    G_model_X = G(bd_model, b_prime_model)
    x = plt.imread(img_path)[:,:,0:3]

    # test_data_filename = str('/content/gdrive/MyDrive/NYU/MLsecurity/CSAW-HackML-2020-BackdoorDetector/data/clean_test_data.h5')
    # test_images, test_labels = data_loader(test_data_filename)
    # test_images= data_preprocess(test_images)
    # x = test_images[1]
    # print(test_labels[1])
    # print(test_images[1][None].shape)

    pred = G_model_X.predict(x[None])
    print(pred)

    # fine_prune.eval(model_path, img_path)
    
    # pass
