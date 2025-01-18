import os
import tensorflow as tf
import keras
# import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.models import Model, load_model
import numpy as np

# import tensorflow_model_optimization as tfmot
keras.backend.clear_session()
# import hls4ml
#os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import gc
gc.collect()
from weights import *


upsampling_factor = {10:(-1,4), 12:(-1,2), 13:(-1,4), 4:(-1,2), 8:(2,-1)}
downsampling_factor ={3:(2,2), 5:(2,1), 6:(2,2),9:(1,2), 14:(1,1), 15:(3,2), 16:(2,1)}

def resample1(in1, in2, Cwt, BN_beta, BN_gamma, BN_var, BN_mean, *args, **kwargs):
    # up and up
    #branch 1
    C01 = Conv2D(use_bias=False, filters = Cwt[0].shape[3], kernel_size= (1,1), weights = [Cwt[0]])
    # C01.set_weights(Cwt[0])
    BN01 = BatchNormalization()
    BN01.beta = BN_beta[0]
    BN01.gamma = BN_gamma[0]
    BN01.moving_mean = BN_mean[0]
    BN01.moving_var = BN_var[0]
    A01 = Activation("relu")

    U01  = UpSampling2D()

    C02 = Conv2D(use_bias=False, filters = Cwt[1].shape[3], kernel_size= (1,1), weights = [Cwt[1]])
    # C02.set_weights(Cwt[1])
    BN02 = BatchNormalization()
    BN02.beta = BN_beta[1]
    BN02.gamma = BN_gamma[1]
    BN02.moving_mean = BN_mean[1]
    BN02.moving_var = BN_var[1]
    A02 = Activation("relu")

    #branch2
    C11 = Conv2D(use_bias=False, filters = Cwt[2].shape[3], kernel_size= (1,1), weights = [Cwt[2]])
    # C11.set_weights(Cwt[2])
    BN11 = BatchNormalization()
    BN11.beta = BN_beta[2]
    BN11.gamma = BN_gamma[2]
    BN11.moving_mean = BN_mean[2]
    BN11.moving_var = BN_var[2]
    A11 = Activation("relu")

    U11  = UpSampling2D()

    C12 = Conv2D(use_bias=False, filters = Cwt[3].shape[3], kernel_size= (1,1), weights = [Cwt[3]])
    # C12.set_weights(Cwt[3])
    BN12 = BatchNormalization()
    BN12.beta = BN_beta[3]
    BN12.gamma = BN_gamma[3]
    BN12.moving_mean = BN_mean[3]
    BN12.moving_var = BN_var[3]
    A12 = Activation("relu")


#B1
    x1 = C01(in1)
    x1 = BN01(x1)
    x1 = A01(x1)

    if kwargs != {}:
        s = upsampling_factor[kwargs['block_no']][0]
        if s>0:
            x1 = UpSampling2D(size = (s,s))(x1)

    x1 = C02(x1)
    x1 = BN02(x1)
    x1 = A02(x1)

#B2
    x2 = C11(in2)
    x2 = BN11(x2)
    x2 = A11(x2)

    # x2 = U11(x2)
    if kwargs != {}:
        s = upsampling_factor[kwargs['block_no']][1]
        if s>0:
            x2 = UpSampling2D(size = (s,s))(x2)

    x2 = C12(x2)
    x2 = BN12(x2)
    x2 = A12(x2)

    Addition = Add()([x1,x2])

    return Activation("relu")(Addition)



def resample2(in1, in2, Cwt, BN_beta, BN_gamma, BN_var, BN_mean, *args, **kwargs):
    #down and down
    #branch 1
    C01 = Conv2D(use_bias=False, filters = Cwt[0].shape[3], kernel_size= (1,1), weights = [Cwt[0]])
    # C01.set_weights(Cwt[0])
    BN01 = BatchNormalization()
    BN01.beta = BN_beta[0]
    BN01.gamma = BN_gamma[0]
    BN01.moving_mean = BN_mean[0]
    BN01.moving_var = BN_var[0]
    A01 = Activation("relu")

    C02 = Conv2D(use_bias=False, filters = Cwt[1].shape[3], kernel_size= (3,3), weights = [Cwt[1]], strides=(2,2), padding = "same")
    # C02.set_weights(Cwt[1])
    BN02 = BatchNormalization()
    BN02.beta = BN_beta[1]
    BN02.gamma = BN_gamma[1]
    BN02.moving_mean = BN_mean[1]
    BN02.moving_var = BN_var[1]
    A02 = Activation("relu")

    C03 = Conv2D(use_bias=False, filters = Cwt[2].shape[3], kernel_size= (1,1), weights = [Cwt[2]])
    # C03.set_weights(Cwt[2])
    BN03 = BatchNormalization()
    BN03.beta = BN_beta[2]
    BN03.gamma = BN_gamma[2]
    BN03.moving_mean = BN_mean[2]
    BN03.moving_var = BN_var[2]
    A03 = Activation("relu")

    #branch 2
    C11 = Conv2D(use_bias=False, filters = Cwt[3].shape[3], kernel_size= (1,1), weights = [Cwt[3]])
    # C11.set_weights(Cwt[3])
    BN11 = BatchNormalization()
    BN11.beta = BN_beta[3]
    BN11.gamma = BN_gamma[3]
    BN11.moving_mean = BN_mean[3]
    BN11.moving_var = BN_var[3]
    A11 = Activation("relu")

    C12 = Conv2D(use_bias=False, filters = Cwt[4].shape[3], kernel_size= (3,3), weights = [Cwt[4]], strides=(2,2), padding = 'same')
    # C12.set_weights(Cwt[4])
    BN12 = BatchNormalization()
    BN12.beta = BN_beta[4]
    BN12.gamma = BN_gamma[4]
    BN12.moving_mean = BN_mean[4]
    BN12.moving_var = BN_var[4]
    A12 = Activation("relu")

    C13 = Conv2D(use_bias=False, filters = Cwt[5].shape[3], kernel_size= (1,1), weights = [Cwt[5]])
    # C13.set_weights(Cwt[5])
    BN13 = BatchNormalization()
    BN13.beta = BN_beta[5]
    BN13.gamma = BN_gamma[5]
    BN13.moving_mean = BN_mean[5]
    BN13.moving_var = BN_var[5]
    A13 = Activation("relu")

        #B1
    x1 = C01(in1)
    x1 = BN01(x1)
    x1 = A01(x1)

    x1 = C02(x1)
    x1 = BN02(x1)
    x1 = A02(x1)
    if kwargs != {}:
        for i in range(downsampling_factor[kwargs['block_no']][0]-1):
            x1 = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x1)

    x1 = C03(x1)
    x1 = BN03(x1)
    x1 = A03(x1)

    #B2
    x2 = C11(in2)
    x2 = BN11(x2)
    x2 = A11(x2)

    x2 = C12(x2)
    x2 = BN12(x2)
    x2 = A12(x2)

    if kwargs != {}:
            for i in range(downsampling_factor[kwargs['block_no']][1]-1):
                x2 = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x2)

    x2 = C13(x2)
    x2 = BN13(x2)
    x2 = A13(x2)

    Addition = Add()([x1,x2])

    return Activation("relu")(Addition)


def resample3(in1, in2, Cwt, BN_beta, BN_gamma, BN_var, BN_mean, *args, **kwargs):
    #up and down

    #branch 1
    C01 = Conv2D(use_bias=False, filters = Cwt[0].shape[3], kernel_size= (1,1), weights = [Cwt[0]])
    # C01.set_weights(Cwt[0])
    BN01 = BatchNormalization()
    BN01.beta = BN_beta[0]
    BN01.gamma = BN_gamma[0]
    BN01.moving_mean = BN_mean[0]
    BN01.moving_var = BN_var[0]
    A01 = Activation("relu")



    C02 = Conv2D(use_bias=False, filters = Cwt[1].shape[3], kernel_size= (1,1), weights = [Cwt[1]])
    # C02.set_weights(Cwt[1])
    BN02 = BatchNormalization()
    BN02.beta = BN_beta[1]
    BN02.gamma = BN_gamma[1]
    BN02.moving_mean = BN_mean[1]
    BN02.moving_var = BN_var[1]
    A02 = Activation("relu")

    #branch 2
    C11 = Conv2D(use_bias=False, filters = Cwt[2].shape[3], kernel_size= (1,1), weights = [Cwt[2]])
    # C11.set_weights(Cwt[2])
    BN11 = BatchNormalization()
    BN11.beta = BN_beta[2]
    BN11.gamma = BN_gamma[2]
    BN11.moving_mean = BN_mean[2]
    BN11.moving_var = BN_var[2]
    A11 = Activation("relu")

    C12 = Conv2D(use_bias=False, filters = Cwt[3].shape[3], kernel_size= (3,3), weights = [Cwt[3]], strides=(2,2), padding = 'same')
    # C12.set_weights(Cwt[3])
    BN12 = BatchNormalization()
    BN12.beta = BN_beta[3]
    BN12.gamma = BN_gamma[3]
    BN12.moving_mean = BN_mean[3]
    BN12.moving_var = BN_var[3]
    A12 = Activation("relu")


    C13 = Conv2D(use_bias=False, filters = Cwt[4].shape[3], kernel_size= (1,1), weights = [Cwt[4]])
    # C13.set_weights(Cwt[4])
    BN13 = BatchNormalization()
    BN13.beta = BN_beta[4]
    BN13.gamma = BN_gamma[4]
    BN13.moving_mean = BN_mean[4]
    BN13.moving_var = BN_var[4]
    A13 = Activation("relu")

    #B1
    x1 = C01(in1)
    x1 = BN01(x1)
    x1 = A01(x1)

    if kwargs != {}:
        if kwargs['block_no']== 4:
            s = upsampling_factor[kwargs['block_no']][1]
            x1 = UpSampling2D(size = (s,s))(x1)

        if kwargs['block_no']== 8:
            s = upsampling_factor[kwargs['block_no']][0]
            x1 = UpSampling2D(size = (s,s))(x1)

    x1 = C02(x1)
    x1 = BN02(x1)
    x1 = A02(x1)

    #B2
    x2 = C11(in2)
    x2 = BN11(x2)
    x2 = A11(x2)

    x2 = C12(x2)
    x2 = BN12(x2)
    x2 = A12(x2)

    x2 = C13(x2)
    x2 = BN13(x2)
    x2 = A13(x2)

    Addition = Add()([x1,x2])

    return Activation("relu")(Addition)


def input_img(in1, Cwt, BN_beta, BN_gamma, BN_var, BN_mean):

    C1 = Conv2D(use_bias=False, filters = 64, kernel_size=(7, 7), weights = [Cwt], padding ='same')
    # C1.set_weights(Cwt)
    BN1 = BatchNormalization()
    BN1.beta = BN_beta
    BN1.gamma = BN_gamma
    BN1.moving_mean = BN_mean
    BN1.moving_var = BN_var
    A1 = Activation("relu")

    x = C1(in1)
    x = BN1(x)
    x = A1(x)

    return x

def output(inp, Cwt, BN_beta, BN_gamma, BN_var, BN_mean):

    C01 = Conv2D(use_bias=False, filters = Cwt.shape[3], kernel_size= (1,1), weights = [Cwt])
    # C01.set_weights(Cwt)
    BN01 = BatchNormalization()
    BN01.beta = BN_beta
    BN01.gamma = BN_gamma
    BN01.moving_mean = BN_mean
    BN01.moving_var = BN_var
    A01 = Activation("relu")

    x1 = C01(inp)
    x1 = BN01(x1)
    x1 = A01(x1)

    return x1

def residual(in1, Cwt, BN_beta, BN_gamma, BN_var, BN_mean):

    C1 = Conv2D(use_bias=False, filters = Cwt[0].shape[3], kernel_size= (3,3), weights = [Cwt[0]], padding = "same")
    # C1.set_weights(Cwt[0])
    BN1 = BatchNormalization()
    BN1.beta = BN_beta[0]
    BN1.gamma = BN_gamma[0]
    BN1.moving_mean = BN_mean[0]
    BN1.moving_var = BN_var[0]
    A1 = Activation("relu")

    C2 = Conv2D(use_bias=False, filters = Cwt[1].shape[3], kernel_size= (3,3), weights = [Cwt[1]], padding = "same")
    # C2.set_weights(Cwt[1])
    BN2 = BatchNormalization()
    BN2.beta = BN_beta[1]
    BN2.gamma = BN_gamma[1]
    BN2.moving_mean = BN_mean[1]
    BN2.moving_var = BN_var[1]
    A2 = Activation("relu")

    x = C1(in1)
    x = BN1(x)
    x = A1(x)

    x = C2(x)
    x = BN2(x)
    x = A2(x)

    return Activation("relu")(x)

def bottleneck(in1, Cwt, BN_beta, BN_gamma, BN_var, BN_mean):

    C1 = Conv2D(use_bias=False, filters = Cwt[0].shape[3], kernel_size= (1,1), weights = [Cwt[0]])
    # C1.set_weights(Cwt[0])
    BN1 = BatchNormalization()
    BN1.beta = BN_beta[0]
    BN1.gamma = BN_gamma[0]
    BN1.moving_mean = BN_mean[0]
    BN1.moving_var = BN_var[0]
    A1 = Activation("relu")

    C2 = Conv2D(use_bias=False, filters = Cwt[1].shape[3], kernel_size= (3,3), weights = [Cwt[1]], padding = "same")
    # C2.set_weights(Cwt[1])
    BN2 = BatchNormalization()
    BN2.beta = BN_beta[1]
    BN2.gamma = BN_gamma[1]
    BN2.moving_mean = BN_mean[1]
    BN2.moving_var = BN_var[1]
    A2 = Activation("relu")

    C3 = Conv2D(use_bias=False, filters = Cwt[2].shape[3], kernel_size= (1,1), weights = [Cwt[2]])
    # C3.set_weights(Cwt[2])
    BN3 = BatchNormalization()
    BN3.beta = BN_beta[2]
    BN3.gamma = BN_gamma[2]
    BN3.moving_mean = BN_mean[2]
    BN3.moving_var = BN_var[2]
    A3 = Activation("relu")

    x = C1(in1)
    x = BN1(x)
    x = A1(x)

    x = C2(x)
    x = BN2(x)
    x = A2(x)

    x = C3(x)
    x = BN3(x)
    x = A3(x)
    return Activation("relu")(x)

# def merge(a,b,c,d,e):

#     res = Concatenate(axis = 2)([a,b,c,d,e])
#     return res
