import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
import matplotlib.pyplot as plt
import keras
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.config.experimental.list_physical_devices('GPU')

clean_data_filename = 'data/cl/valid.h5'
poisoned_data_filename = 'data/bd/bd_valid.h5'
clean_test_filename = 'data/cl/test.h5'
poisoned_test_filename = 'data/bd/bd_test.h5'
model_filename = 'models/bd_net.h5'

def prune_defence(model):
    # get the clean and poisoned data
    cl_x_test, cl_y_test = utils.data_loader(clean_data_filename)
    bd_x_test, bd_y_test = utils.data_loader(poisoned_data_filename)

    clean_data_acc = 98.64899974019225  # original accuracy
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    # prune_index = []
    clean_acc = []
    asrate = []
    saved_model = [False for _ in range(3)]

    # get the activation from 'pool_3'
    layer_output = model_copy.get_layer('pool_3').output  # output of 'pool_3'
    intermediate_model = keras.models.Model(inputs=model_copy.input, outputs=layer_output)
    intermediate_prediction = intermediate_model.predict(cl_x_test)  # activation value
    temp = np.mean(intermediate_prediction, axis=(0, 1, 2))  # average activation value
    seq = np.argsort(temp)
    weight_0 = model_copy.layers[5].get_weights()[0]  # 'conv_3', size=(3, 3, 40, 60)
    bias_0 = model_copy.layers[5].get_weights()[1]  # 'conv_3', size=(60, 60)

    for channel_index in tqdm(seq):
        weight_0[:, :, :, channel_index] = 0  # prune 'conv_3'
        bias_0[channel_index] = 0
        model_copy.layers[5].set_weights([weight_0, bias_0])
        cl_label_p = np.argmax(model_copy.predict(cl_x_test), axis=1)
        clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
        saved_model_action = saved_model.copy()
        if (clean_data_acc - clean_accuracy >= 2 and not saved_model[0]):
            print("The accuracy drops at least 2%, saved the model")
            model_copy.save('model_B_2.h5')
            saved_model[0] = True
        if (clean_data_acc - clean_accuracy >= 4 and not saved_model[1]):
            print("The accuracy drops at least 4%, saved the model")
            model_copy.save('model_B_4.h5')
            saved_model[1] = True
        if (clean_data_acc - clean_accuracy >= 10 and not saved_model[2]):
            print("The accuracy drops at least 10%, saved the model")
            model_copy.save('model_B_10.h5')
            saved_model[2] = True
        clean_acc.append(clean_accuracy)
        bd_label_p = np.argmax(model_copy.predict(bd_x_test), axis=1)
        asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
        asrate.append(asr)
        if saved_model_action != saved_model:
            print("\nThe clean accuracy is: ", clean_accuracy)
            print("The attack success rate is: ", asr)
        # print("  The pruned channel index is: ", channel_index)
        keras.backend.clear_session()

    # plot prune defence result
    x_axis = np.arange(1, 61) / 60
    plt.plot(x_axis, clean_acc)
    plt.plot(x_axis, asrate)
    for idx, i in enumerate(asrate):
        if i <= 1:
            plt.scatter((idx+1)/60.0, i, s=20, marker='x')
            plt.text((idx+1)/60.0, i, (float('%.2f' % ((idx+1)/60.0)), float('%.2f' % i)), ha='left', va='top')
            plt.scatter((idx + 1) / 60.0, clean_acc[idx], s=20, marker='x')
            plt.text((idx + 1) / 60.0, clean_acc[idx], (float('%.2f' % ((idx + 1) / 60.0)), float('%.2f' % clean_acc[idx])), ha='left', va='top')
            break
    plt.legend(['clean_accuracy', 'attack success rate'])
    plt.xlabel("fraction of pruned channels")
    plt.ylabel("rate")
    plt.title("accuracy and attack success rate for validation dataset")
    plt.savefig('prune_result.png')
    plt.show()

class G(keras.Model):  # GoodNet
  def __init__(self, B, B_prime):
      super(G, self).__init__()
      self.B = B
      self.B_prime = B_prime

  def predict(self, data):
      y = np.argmax(self.B(data), axis=1)
      y_prime = np.argmax(self.B_prime(data), axis=1)
      pred = np.zeros(data.shape[0])
      for i in range(data.shape[0]):
        if y[i]==y_prime[i]:
          pred[i] = y[i]
        else:
          pred[i] = 1283  # total class: 1282
      return pred

def eval_defence():
    cl_x_test, cl_y_test = utils.data_loader(clean_test_filename)
    bd_x_test, bd_y_test = utils.data_loader(poisoned_test_filename)
    bd_model = keras.models.load_model(model_filename)

    Bprime_model_2 = keras.models.load_model('model_B_2.h5', compile=False)
    Bprime_model_4 = keras.models.load_model('model_B_4.h5', compile=False)
    Bprime_model_10 = keras.models.load_model('model_B_10.h5', compile=False)

    # evaluate with the model
    G_model_X_2 = G(bd_model, Bprime_model_2)
    G_model_X_4 = G(bd_model, Bprime_model_4)
    G_model_X_10 = G(bd_model, Bprime_model_10)

    G_clean_test_2_accuracy = np.mean(np.equal(G_model_X_2.predict(cl_x_test), cl_y_test)) * 100
    G_asr_2 = np.mean(np.equal(G_model_X_2.predict(bd_x_test), bd_y_test)) * 100

    G_clean_test_4_accuracy = np.mean(np.equal(G_model_X_4.predict(cl_x_test), cl_y_test)) * 100
    G_asr_4 = np.mean(np.equal(G_model_X_4.predict(bd_x_test), bd_y_test)) * 100

    G_clean_test_10_accuracy = np.mean(np.equal(G_model_X_10.predict(cl_x_test), cl_y_test)) * 100
    G_asr_10 = np.mean(np.equal(G_model_X_10.predict(bd_x_test), bd_y_test)) * 100

    G_data = {
        "GoodNet_G_model": ["acc_drop_2%", "acc_drop_4%", "acc_drop_10%"],
        "clean_test_acc": [G_clean_test_2_accuracy,
                       G_clean_test_4_accuracy,
                       G_clean_test_10_accuracy],
        "bad_attack_rate": [G_asr_2, G_asr_4, G_asr_10]
    }
    G_df = pd.DataFrame(G_data)
    G_df.set_index('GoodNet_G_model')
    print(G_df.to_string(index=False))

if __name__ == '__main__':
    keras.backend.clear_session()

    model = utils.load_model(model_filename)
    prune_defence(model)

    eval_defence()
