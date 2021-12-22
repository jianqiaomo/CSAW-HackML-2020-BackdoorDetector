import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import strip
import sys
import keras

if __name__ == '__main__':
    model_path = ['../models/anonymous_1_bd_net.h5', '../models/anonymous_2_bd_net.h5',
                  '../models/multi_trigger_multi_target_bd_net.h5', '../models/sunglasses_bd_net.h5']
    save_repaire_path = ['../models_G/anonymous_1_STRIP_net.h5', '../models_G/anonymous_2_STRIP_net.h5',
                  '../models_G/multi_trigger_multi_target_STRIP_net.h5', '../models_G/sunglasses_STRIP_net.h5']

    clean_path = '../data/clean_validation_data.h5'

    for i, model_p in enumerate(model_path):
        STRIP_net = strip.RepairedNetG(keras.models.load_model(model_p), clean_path)
        STRIP_net.save(save_repaire_path[i])
        print("save to: ", save_repaire_path[i])
