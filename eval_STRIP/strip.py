import os
import sys
import random
import h5py
import numpy as np
import keras
from tqdm import tqdm
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# # convert img value range
# # from (0, 255) -> (0, 1)
# def preprocess(x):
#     return x / 255.0


# # show imgs in sorted dataset D with class in cls_list
# def show(D, cls_list):
#     n = len(cls_list)
#     plt.figure(figsize=(2*n,n))
#     for j, cls in enumerate(cls_list):
#         for i in range(9):
#             plt.subplot(9, 2*n, j*2*n+i+1)
#             plt.imshow(D[cls, i])








# def entropy(logit):
#     prob = logit / np.sum(logit)
#     sum = 0
#     for p in prob:
#         if p == 0: item = 0
#         else: item = - p * np.log2(p)
#         sum += item
#     return sum


# # judge whether a single input x is a trojaned input
# # also return H
# def detect_trojan(model, x, D, boundary):
#     N = len(D) # num_class
#     H = 0
#     perturbed_x = perturbation_step(x, D, 0.5)
#     logits = model.predict(perturbed_x)
#     for logit in logits:
#         H_i = entropy(logit)
#         H += H_i
#     H /= N
#
#     if H <= boundary:
#         pred = N
#     else:
#         logit = model.predict(np.expand_dims(x, axis=0))
#         pred = np.argmax(logit, axis=1)
#
#     return pred, H








# def G(model_path, data_path):
#     num_class = 1283
#
#     valid_data_path = 'data/clean_validation_data.h5'
#
#     xval, yval = load_data(valid_data_path)
#     xval = preprocess(xval)
#     D = sort_samples(xval, yval, num_class)
#
#     badnet = load_model(model_path)
#     x, y = load_data(data_path)
#     x = preprocess(x)
#
#     pred = detect_trojan_batch(badnet, x, D, 0.5)
#
#     return pred


# def eval(model_path, img_path):
#
#     xval, yval = load_data('../data/clean_validation_data.h5')
#     xval = preprocess(xval)
#     D = sort_samples(xval, yval, 1283)
#
#     badnet = load_model(model_path)
#     x = plt.imread(img_path)[:,:,0:3]
#
#     pred = detect_trojan(badnet, x, D, 0.5)
#
#     print(pred[0])

class RepairedNetG(keras.Model):
    def __init__(self, BadNet, CleanData=None, boundary=0.5):
        super(RepairedNetG, self).__init__()
        self.B = BadNet
        self.CleanData = CleanData
        self.boundary = boundary

    # load data from h5 file
    def load_data(self, file_path):
        data = h5py.File(file_path, 'r')
        x_data = np.array(data['data'])
        y_data = np.array(data['label'])
        x_data = x_data.transpose((0, 2, 3, 1))
        y_data = to_categorical(y_data, num_classes=1283)
        return x_data, y_data

    def load_clean_data(self):
        if self.CleanData is not None:
            # self.xval, self.yval = self.load_data('../data/clean_validation_data.h5')
            self.xval, self.yval = self.load_data(self.CleanData)
            # preprocess: convert img value range from (0, 255) -> (0, 1)
            self.xval = self.xval / 255.0

    def detect_trojan(self, x):
        N = len(self.D)  # num_class
        H = 0
        perturbed_x = self.perturbation_step(x, self.D, self.boundary)
        logits = self.B.predict(perturbed_x)
        for logit in logits:
            H_i = self.entropy(logit)
            H += H_i
        H /= N

        if H <= self.boundary:
            pred = N
        else:
            logit = self.B.predict(np.expand_dims(x, axis=0))
            pred = np.argmax(logit, axis=1)
        return pred, H

    def entropy_layer(self):
        self.load_clean_data()
        self.D = self.sort_samples(self.xval, self.yval, 1283)

    def predict_img(self, img_path):
        self.entropy_layer()
        x = plt.imread(img_path)[:,:,0:3]
        return self.detect_trojan(x)  # (pred, H)

    def predict(self, data_x):
        self.entropy_layer()
        # x = plt.imread(img_path)[:,:,0:3]
        return self.detect_trojan_batch(data_x)  # (pred[], H[])

    def save(self, filename):
        self.B.save(filename)

    # sort the clean sample dataset
    def sort_samples(self, xval, yval, num_classes):
        """
        :param xval: validation data
        :type xval: np.array, (N, H, W, C), N=num_classes*9, H=55, W=47, C=3. imgs
        :param yval: validation data label corresponding to xval
        :type yval: np.array, (N, num_classes), num_classes=1283, logits
        :param num_classes: number of classes of the dataset
        :type num_classes: int
        :returns: sorted xval, each classes with its own list
        :rtype: np.array, (num_classes, 9, H, W, C)
        """

        index = np.zeros((1283, 9)).astype('int')
        cls_cnt = np.zeros(1283).astype('int')
        yval = np.argmax(yval, axis=1)
        for i in range(xval.shape[0]):
            # index[c, x], index of the x-th img in class c in xval
            index[yval[i], cls_cnt[yval[i]]] = i
            cls_cnt[yval[i]] += 1

        D = []
        for cls in range(num_classes):
            cls_samples = []
            for i in range(9):
                cls_samples.append(self.xval[index[cls, i]])
            # append a new list with 9 same classes imgs
            D.append(cls_samples)
        D = np.array(D)

        return D

    def entropy(self, logit):
        prob = logit / np.sum(logit)
        sum = 0
        for p in prob:
            if p == 0:
                item = 0
            else:
                item = - p * np.log2(p)
            sum += item
        return sum

    # blend two imgs
    def perturbe(self, b, t, alpha):
        assert alpha < 1 and alpha > 0, "r must satisfy 0 < r < 1"
        return b * alpha + t * (1 - alpha)

    # genarate perturbed inputs for BadNet
    def perturbation_step(self, x, D, alpha):
        assert alpha < 1 and alpha > 0, "r must satisfy 0 < r < 1"
        N = len(D)  # num_class
        perturbed_x = []
        for i in range(N):
            a = random.randint(0, N - 1)  # class
            b = random.randint(0, 8)  # index in the class list
            perturbed_x.append(x)
            perturbed_x[i] = self.perturbe(perturbed_x[i], D[a, b], alpha)
        perturbed_x = np.array(perturbed_x)
        return perturbed_x

    def perturbation_step_batch(self, x, D, alpha):
        nsamp = x.shape[0] # 12830
        N = len(D)
        perturbed_inputs = []
        # replicas = np.expand_dims(x, 1).repeat(9, axis=1)
        for x_i in tqdm(range(nsamp)):
            perturbed_x = []
            for ii in range(N):
                i = np.random.randint(0, N - 1)  # i = np.random.randint(0, N - 1, size=[nsamp, 9])
                cls = np.random.randint(0, 8)  # cls = np.random.randint(0, 8, size=[nsamp, 9])
                random_test_samples = D[i, cls]
                perturbed_x.append(self.perturbe(x[x_i], random_test_samples, alpha))
            perturbed_inputs.append(np.array(perturbed_x))
        return perturbed_inputs

        # perturbed_inputs = []
        # for x_i in x:
        #     perturbed_inputs.append(self.perturbation_step(x_i, D, alpha))

    # def perturbation_step_batch(self, x, D, alpha):
    #     nsamp = x.shape[0]
    #     N = len(D)
    #     replicas = np.expand_dims(x, 1).repeat(9, axis=1)
    #     i = np.random.randint(0, N - 1, size=[nsamp, 9])
    #     cls = np.random.randint(0, 8, size=[nsamp, 9])
    #     random_test_samples = D[i, cls]
    #     perturbed_inputs = self.perturbe(replicas, random_test_samples, alpha)
    #     return perturbed_inputs

    # def entropy_batch(self, logits):
    #     H_n = []
    #     for logit in logits:
    #         prob = logit / np.sum(logit)
    #         sum = 0
    #         for p in prob:
    #             if p == 0:
    #                 item = 0
    #             else:
    #                 item = - p * np.log2(p)
    #             sum += item
    #         H_n.append(sum)
    #     H_n = np.array(H_n)
    #     return H_n

    def entropy_batch(self, logits, N):
        H = 0
        for logit in logits:
            H_i = self.entropy(logit)
            H += H_i
        H /= N
        return H

    # def detect_trojan_batch(self, x):
    #     N = len(self.D)
    #     pred = []
    #
    #     perturbed_inputs = self.perturbation_step_batch(x, self.D, self.boundary)
    #
    #     H_list = []
    #     for i in tqdm(range(len(perturbed_inputs))):  # for i in tqdm(range(N)):
    #         perturbed_input = perturbed_inputs[i]
    #         logits = self.B.predict(perturbed_input)
    #         H_n = self.entropy_batch(logits, N)
    #
    #         H_list.append(H_n)
    #         if (H_n < self.boundary):
    #             pred.append(N)
    #         else:
    #             logit = self.B.predict(np.expand_dims(perturbed_input, axis=0))
    #             pred.append(np.argmax(logit, axis=1))
    #     H_list = np.array(H_list)
    #     return pred, H_list

    def detect_trojan_batch(self, x):
        pred = []
        H_list = []

        for x_i in tqdm(x):
            pred_i, H_list_i = self.detect_trojan(x_i)
            pred.append(pred_i)
            H_list.append(H_list_i)

        return pred, H_list