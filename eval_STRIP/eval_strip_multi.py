import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import strip
import sys
import keras
import argparse
from lab3.utils import data_loader
import numpy as np
import random

def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Compiler and Simulator args')
    parser.add_argument('img_path')
    parser.add_argument('--testset', dest='test',
                        help='test with test_set',
                        type=str, default=None)
    parser.add_argument('--sample_num', dest='sample_num',
                        help='test sample number',
                        type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    model_path = '../models_G/multi_trigger_multi_target_STRIP_net.h5'  # '../models/anonymous_1_bd_net.h5'
    parser = argparse.ArgumentParser(description='img_path')

    img_path = args.img_path  # = str(sys.argv[1])
    clean_path = '../data/clean_validation_data.h5'

    STRIP_net = keras.models.load_model(model_path)
    if args.test is None:  # test one image (img_path)
        print("load: ", model_path)
        G1 = strip.RepairedNetG(STRIP_net, '../data/clean_validation_data.h5')
        print(G1.predict_img(img_path)[0])  # output [0, 1283]
    else:  # test with a test_set
        sample_num = args.sample_num
        test_x, test_y = data_loader(args.test, False, True)
        G1 = strip.RepairedNetG(STRIP_net, '../data/clean_validation_data.h5')
        start_point = random.randint(0, len(test_y) - sample_num - 1)
        G1_pred = G1.predict(test_x[start_point:start_point + sample_num, :, :, :])[0]
        pred = [int(i) for i in G1_pred]
        match_rate = np.mean(np.equal(pred, test_y[:sample_num]))
        print("match_rate: ", match_rate)


