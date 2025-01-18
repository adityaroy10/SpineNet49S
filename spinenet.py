import os
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.models import Model, load_model
import numpy as np
keras.backend.clear_session()
import matplotlib.pyplot as plt
import cv2
import gc
gc.collect()

from weights import *
from spinenet_functions import *

def SpineNet():

    input = Input(shape=(640,640,3))
    x = input_img(input, in0_conv, in0_beta, in0_gamma, in0_var, in0_mean)
    # sc = x
    print("before max pool 1",x.shape)
    x00 = MaxPooling2D(pool_size = (3,3),strides = (2,2), padding = "same")(x)
    print("after max pool 2",x.shape)

    x01 = bottleneck(x00, block0_conv, block0_beta, block0_gamma, block0_var, block0_mean)
    print("after stem 0",x01.shape)

    x02 = bottleneck(x01, block1_conv, block1_beta, block1_gamma, block1_var, block1_mean)
    print("after stem 1",x02.shape)


    #Skip Connection
    C = Conv2D(use_bias=False, filters = 164, kernel_size= (1,1), weights = [in1_conv])
    # C3.set_weights(Cwt[2])
    BN = BatchNormalization()
    BN.beta = in1_beta
    BN.gamma = in1_gamma
    BN.moving_mean = in1_mean
    BN.moving_var = in1_var
    A = Activation("relu")

    SC = C(x00)
    SC = BN(SC)
    # SC = A(SC)

    x02 = Add()([x02, SC])

    x02 = A(x02)


    in1 = resample1(x01, x02, block2_in_conv, block2_in_beta, block2_in_gamma, block2_in_var, block2_in_mean)
    x1 = bottleneck(in1, block2_conv, block2_beta, block2_gamma, block2_var, block2_mean)
    print("after block 2",x1.shape)

    in2 = resample2(x01, x02, block3_in_conv, block3_in_beta, block3_in_gamma, block3_in_var, block3_in_mean, block_no = 3)
    x2 = residual(in2, block3_conv, block3_beta, block3_gamma, block3_var, block3_mean)
    print("after block 3",x2.shape)

    in3 = resample3(x2, x1, block4_in_conv, block4_in_beta, block4_in_gamma, block4_in_var, block4_in_mean, block_no = 4)
    x3 = bottleneck(in3, block4_conv, block4_beta, block4_gamma, block4_var, block4_mean)
    print("after block 4",x3.shape)

    in4 = resample2(x1, x3, block5_in_conv, block5_in_beta, block5_in_gamma, block5_in_var, block5_in_mean, block_no = 5)
    x4 = bottleneck(in4, block5_conv, block5_beta, block5_gamma, block5_var, block5_mean)
    print("after block 5",x4.shape)

    in5 = resample2(x2, x4, block6_in_conv, block6_in_beta, block6_in_gamma, block6_in_var, block6_in_mean, block_no = 6)
    x5 = residual(in5, block6_conv, block6_beta, block6_gamma, block6_var, block6_mean)
    print("after block 6",x5.shape)

    in6 = resample1(x2, x4, block7_in_conv, block7_in_beta, block7_in_gamma, block7_in_var, block7_in_mean)
    x6 = bottleneck(in6, block7_conv, block7_beta, block7_gamma, block7_var, block7_mean)
    print("after block 7", x6.shape)

    in7 = resample3(x5, x6, block8_in_conv, block8_in_beta, block8_in_gamma, block8_in_var, block8_in_mean, block_no = 8)
    x7 = residual(in7, block8_conv, block8_beta, block8_gamma, block8_var, block8_mean)
    print("after block 8",x7.shape)

    in8 = resample2(x5, x7, block9_in_conv, block9_in_beta, block9_in_gamma, block9_in_var, block9_in_mean, block_no = 9)
    x8 = residual(in8, block9_conv, block9_beta, block9_gamma, block9_var, block9_mean)
    print("after block 9",x8.shape)

    in9 = resample1(x7, x8, block10_in_conv, block10_in_beta, block10_in_gamma, block10_in_var, block10_in_mean, block_no = 10)
    x9 = bottleneck(in9, block10_conv, block10_beta, block10_gamma, block10_var, block10_mean)
    print("after block 10",x9.shape)

    in10 = resample1(x7, x9, block11_in_conv, block11_in_beta, block11_in_gamma, block11_in_var, block11_in_mean)
    x10 = bottleneck(in10, block11_conv, block11_beta, block11_gamma, block11_var, block11_mean)
    print("after block 11",x10.shape)

    # output blocks
    in11 = resample1(x4, x9, block12_in_conv, block12_in_beta, block12_in_gamma, block12_in_var, block12_in_mean, block_no = 12)
    x11 = bottleneck(in11, block12_conv, block12_beta, block12_gamma, block12_var, block12_mean)
    o1 = output(x11, out_conv_12, out_BN12_beta, out_BN12_gamma, out_BN12_var, out_BN12_mean)
    print("after block 12",x11.shape)
    print('output 12', o1.shape)

    in12 = resample1(x3, x9, block13_in_conv, block13_in_beta, block13_in_gamma, block13_in_var, block13_in_mean, block_no = 13)
    x12 = bottleneck(in12, block13_conv, block13_beta, block13_gamma, block13_var, block13_mean)
    o2 = output(x12, out_conv_13, out_BN13_beta, out_BN13_gamma, out_BN13_var, out_BN14_mean)
    print("after block 13",x12.shape)
    print('output 13', o2.shape)

    ######### Third Input ###########
    in13 = resample2(x6, x11, block14_in_conv, block14_in_beta, block14_in_gamma, block14_in_var, block14_in_mean, block_no = 14)
    in13 = Add()([in13, x10])
    x13 = bottleneck(in13, block14_conv, block14_beta, block14_gamma, block14_var, block14_mean)
    o3 = output(x13, out_conv_14, out_BN14_beta, out_BN14_gamma, out_BN14_var, out_BN14_mean)
    print("after block 14",x13.shape)
    print('output 14', o3.shape)

    in14 = resample2(x4, x13, block15_in_conv, block15_in_beta, block15_in_gamma, block15_in_var, block15_in_mean, block_no = 15)
    x14 = bottleneck(in14, block15_conv, block15_beta, block15_gamma, block15_var, block15_mean)
    o4 = output(x14, out_conv_15, out_BN15_beta, out_BN15_gamma, out_BN15_var, out_BN15_mean)
    print("after block 15",x14.shape)
    print('output 15', o4.shape)

    in15 = resample2(x11, x13, block16_in_conv, block16_in_beta, block16_in_gamma, block16_in_var, block16_in_mean, block_no = 16)
    x15 = bottleneck(in15, block16_conv, block16_beta, block16_gamma, block16_var, block16_mean)
    o5 = output(x15, out_conv_16, out_BN16_beta, out_BN16_gamma, out_BN16_var, out_BN16_mean)
    print("after block 16",x15.shape)
    print('output 16', o5.shape)


    model = keras.Model(inputs = input, outputs = [o1, o2, o3, o4, o5], name = 'spinenet')

    return model
