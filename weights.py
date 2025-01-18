PATH_TO_CKPT_DIR = "./models/SpineNet-49S"
# PATH_TO_CKPT_DIR = "d:\ICTP\SN49S\models\SpineNet-49S"
import os
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.models import Model, load_model
import numpy as np

reader = tf.train.load_checkpoint(PATH_TO_CKPT_DIR)
shape = reader.get_variable_to_shape_map()
dtype = reader.get_variable_to_dtype_map()
keys = shape.keys()
weights ={}
for key in keys:
    weights[key]=reader.get_tensor(key)

#input convolutions

in0_conv = weights["spinenet/conv2d/kernel"]
in1_conv = weights["spinenet/conv2d_1/kernel"]

in0_beta = weights["spinenet/batch_normalization/beta"]
in1_beta = weights["spinenet/batch_normalization_1/beta"]

in0_gamma = weights["spinenet/batch_normalization/gamma"]
in1_gamma = weights["spinenet/batch_normalization_1/gamma"]

in0_var = weights["spinenet/batch_normalization/moving_variance"]
in1_var = weights["spinenet/batch_normalization_1/moving_variance"]

in0_mean = weights["spinenet/batch_normalization/moving_mean"]
in1_mean = weights["spinenet/batch_normalization_1/moving_mean"]

#output convolutions
out_conv_13 = weights["spinenet/conv2d_8/kernel"]
out_conv_12 = weights["spinenet/conv2d_9/kernel"]
out_conv_14 = weights["spinenet/conv2d_10/kernel"]
out_conv_16 = weights["spinenet/conv2d_11/kernel"]
out_conv_15 = weights["spinenet/conv2d_12/kernel"]

out_BN13_gamma = weights["spinenet/batch_normalization_8/gamma"]
out_BN12_gamma = weights["spinenet/batch_normalization_9/gamma"]
out_BN14_gamma = weights["spinenet/batch_normalization_10/gamma"]
out_BN16_gamma = weights["spinenet/batch_normalization_11/gamma"]
out_BN15_gamma = weights["spinenet/batch_normalization_12/gamma"]

out_BN13_beta = weights["spinenet/batch_normalization_8/beta"]
out_BN12_beta = weights["spinenet/batch_normalization_9/beta"]
out_BN14_beta = weights["spinenet/batch_normalization_10/beta"]
out_BN16_beta = weights["spinenet/batch_normalization_11/beta"]
out_BN15_beta = weights["spinenet/batch_normalization_12/beta"]

out_BN13_var = weights["spinenet/batch_normalization_8/moving_variance"]
out_BN12_var = weights["spinenet/batch_normalization_9/moving_variance"]
out_BN14_var = weights["spinenet/batch_normalization_10/moving_variance"]
out_BN16_var = weights["spinenet/batch_normalization_11/moving_variance"]
out_BN15_var = weights["spinenet/batch_normalization_12/moving_variance"]

out_BN13_mean = weights["spinenet/batch_normalization_8/moving_mean"]
out_BN12_mean = weights["spinenet/batch_normalization_9/moving_mean"]
out_BN14_mean = weights["spinenet/batch_normalization_10/moving_mean"]
out_BN16_mean = weights["spinenet/batch_normalization_11/moving_mean"]
out_BN15_mean = weights["spinenet/batch_normalization_12/moving_mean"]

# Internal block weights

#block 0
conv1 = weights["spinenet/conv2d_2/kernel"]
conv2 = weights["spinenet/conv2d_3/kernel"]
conv3 = weights["spinenet/conv2d_4/kernel"]

beta1 = weights["spinenet/batch_normalization_2/beta"]
beta2 = weights["spinenet/batch_normalization_3/beta"]
beta3 = weights["spinenet/batch_normalization_4/beta"]

gamma1 = weights["spinenet/batch_normalization_2/gamma"]
gamma2 = weights["spinenet/batch_normalization_3/gamma"]
gamma3 = weights["spinenet/batch_normalization_4/gamma"]

var1 = weights["spinenet/batch_normalization_2/moving_variance"]
var2 = weights["spinenet/batch_normalization_3/moving_variance"]
var3 = weights["spinenet/batch_normalization_4/moving_variance"]

mean1 = weights["spinenet/batch_normalization_2/moving_mean"]
mean2 = weights["spinenet/batch_normalization_3/moving_mean"]
mean3 = weights["spinenet/batch_normalization_4/moving_mean"]

block0_conv = [conv1, conv2, conv3]
block0_beta = [beta1, beta2, beta3]
block0_mean = [mean1, mean2, mean3]
block0_var =  [var1, var2, var3]
block0_gamma = [gamma1, gamma2, gamma3]

#block 1
conv1 = weights["spinenet/conv2d_5/kernel"]
conv2 = weights["spinenet/conv2d_6/kernel"]
conv3 = weights["spinenet/conv2d_7/kernel"]

beta1 = weights["spinenet/batch_normalization_5/beta"]
beta2 = weights["spinenet/batch_normalization_6/beta"]
beta3 = weights["spinenet/batch_normalization_7/beta"]

gamma1 = weights["spinenet/batch_normalization_5/gamma"]
gamma2 = weights["spinenet/batch_normalization_6/gamma"]
gamma3 = weights["spinenet/batch_normalization_7/gamma"]

var1 = weights["spinenet/batch_normalization_5/moving_variance"]
var2 = weights["spinenet/batch_normalization_6/moving_variance"]
var3 = weights["spinenet/batch_normalization_7/moving_variance"]

mean1 = weights["spinenet/batch_normalization_5/moving_mean"]
mean2 = weights["spinenet/batch_normalization_6/moving_mean"]
mean3 = weights["spinenet/batch_normalization_7/moving_mean"]

block1_conv = [conv1, conv2, conv3]
block1_beta = [beta1, beta2, beta3]
block1_mean = [mean1, mean2, mean3]
block1_var =  [var1, var2, var3]
block1_gamma = [gamma1, gamma2, gamma3]

#block 2
i=2
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block2_conv = [conv1, conv2, conv3]
block2_beta = [beta1, beta2, beta3]
block2_mean = [mean1, mean2, mean3]
block2_var =  [var1, var2, var3]
block2_gamma = [gamma1, gamma2, gamma3]


#block 3
i=3
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]

block3_conv = [conv1, conv2]
block3_beta = [beta1, beta2]
block3_mean = [mean1, mean2]
block3_var =  [var1, var2]
block3_gamma = [gamma1, gamma2]


#block 4
i=4
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block4_conv = [conv1, conv2, conv3]
block4_beta = [beta1, beta2, beta3]
block4_mean = [mean1, mean2, mean3]
block4_var =  [var1, var2, var3]
block4_gamma = [gamma1, gamma2, gamma3]

#block 5
i=5
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block5_conv = [conv1, conv2, conv3]
block5_beta = [beta1, beta2, beta3]
block5_mean = [mean1, mean2, mean3]
block5_var =  [var1, var2, var3]
block5_gamma = [gamma1, gamma2, gamma3]

#block 6
i=6
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]

block6_conv = [conv1, conv2]
block6_beta = [beta1, beta2]
block6_mean = [mean1, mean2]
block6_var =  [var1, var2]
block6_gamma = [gamma1, gamma2]

#block 7
i=7
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block7_conv = [conv1, conv2, conv3]
block7_beta = [beta1, beta2, beta3]
block7_mean = [mean1, mean2, mean3]
block7_var =  [var1, var2, var3]
block7_gamma = [gamma1, gamma2, gamma3]

#block 8
i=8
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]

block8_conv = [conv1, conv2]
block8_beta = [beta1, beta2]
block8_mean = [mean1, mean2]
block8_var =  [var1, var2]
block8_gamma = [gamma1, gamma2]

#block 9
i=9
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]

block9_conv = [conv1, conv2]
block9_beta = [beta1, beta2]
block9_mean = [mean1, mean2]
block9_var =  [var1, var2]
block9_gamma = [gamma1, gamma2]


#block 10
i=10
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block10_conv = [conv1, conv2, conv3]
block10_beta = [beta1, beta2, beta3]
block10_mean = [mean1, mean2, mean3]
block10_var =  [var1, var2, var3]
block10_gamma = [gamma1, gamma2, gamma3]

#block 11
i=11
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block11_conv = [conv1, conv2, conv3]
block11_beta = [beta1, beta2, beta3]
block11_mean = [mean1, mean2, mean3]
block11_var =  [var1, var2, var3]
block11_gamma = [gamma1, gamma2, gamma3]

#block 12
i=12
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block12_conv = [conv1, conv2, conv3]
block12_beta = [beta1, beta2, beta3]
block12_mean = [mean1, mean2, mean3]
block12_var =  [var1, var2, var3]
block12_gamma = [gamma1, gamma2, gamma3]

#block 13
i=13
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block13_conv = [conv1, conv2, conv3]
block13_beta = [beta1, beta2, beta3]
block13_mean = [mean1, mean2, mean3]
block13_var =  [var1, var2, var3]
block13_gamma = [gamma1, gamma2, gamma3]

#block 14
i=14
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block14_conv = [conv1, conv2, conv3]
block14_beta = [beta1, beta2, beta3]
block14_mean = [mean1, mean2, mean3]
block14_var =  [var1, var2, var3]
block14_gamma = [gamma1, gamma2, gamma3]

#block 15
i=15
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block15_conv = [conv1, conv2, conv3]
block15_beta = [beta1, beta2, beta3]
block15_mean = [mean1, mean2, mean3]
block15_var =  [var1, var2, var3]
block15_gamma = [gamma1, gamma2, gamma3]


#block 16
i=16
conv1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/scale_permuted_block_{i}/batch_normalization_2/moving_mean"]

block16_conv = [conv1, conv2, conv3]
block16_beta = [beta1, beta2, beta3]
block16_gamma = [gamma1, gamma2, gamma3]
block16_mean = [mean1, mean2, mean3]
block16_var =  [var1, var2, var3]

#Resampling weights for blocks

#block 2
i = 2
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block2_in_conv = [conv1, conv2, conv3, conv4]
block2_in_beta = [beta1, beta2, beta3, beta4]
block2_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block2_in_var = [var1, var2, var3, var4]
block2_in_mean = [mean1, mean2, mean3, mean4]

#block 3
i = 3
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block3_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block3_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block3_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block3_in_var = [var1, var2, var3, var4, var5, var6]
block3_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]

#block 4
i = 4
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]

block4_in_conv = [conv1, conv2, conv3, conv4, conv5]
block4_in_beta = [beta1, beta2, beta3, beta4, beta5]
block4_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5]
block4_in_var = [var1, var2, var3, var4, var5]
block4_in_mean = [mean1, mean2, mean3, mean4, mean5]

#block 5
i = 5
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block5_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block5_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block5_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block5_in_var = [var1, var2, var3, var4, var5, var6]
block5_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]

#block 6
i = 6
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block6_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block6_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block6_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block6_in_var = [var1, var2, var3, var4, var5, var6]
block6_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]

#block 7
i = 7
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block7_in_conv = [conv1, conv2, conv3, conv4]
block7_in_beta = [beta1, beta2, beta3, beta4]
block7_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block7_in_var = [var1, var2, var3, var4]
block7_in_mean = [mean1, mean2, mean3, mean4]

#block 8
i = 8
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block8_in_conv = [conv1, conv2, conv3, conv4, conv5]
block8_in_beta = [beta1, beta2, beta3, beta4, beta5]
block8_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5]
block8_in_var = [var1, var2, var3, var4, var5]
block8_in_mean = [mean1, mean2, mean3, mean4, mean5]

#block 9
i = 9
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block9_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block9_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block9_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block9_in_var = [var1, var2, var3, var4, var5, var6]
block9_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]


#block 10
i = 10
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block10_in_conv = [conv1, conv2, conv3, conv4]
block10_in_beta = [beta1, beta2, beta3, beta4]
block10_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block10_in_var = [var1, var2, var3, var4]
block10_in_mean = [mean1, mean2, mean3, mean4]


#block 11
i = 11
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block11_in_conv = [conv1, conv2, conv3, conv4]
block11_in_beta = [beta1, beta2, beta3, beta4]
block11_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block11_in_var = [var1, var2, var3, var4]
block11_in_mean = [mean1, mean2, mean3, mean4]

#block 12
i = 12
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block12_in_conv = [conv1, conv2, conv3, conv4]
block12_in_beta = [beta1, beta2, beta3, beta4]
block12_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block12_in_var = [var1, var2, var3, var4]
block12_in_mean = [mean1, mean2, mean3, mean4]

#block 13
i = 13
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]

block13_in_conv = [conv1, conv2, conv3, conv4]
block13_in_beta = [beta1, beta2, beta3, beta4]
block13_in_gamma = [gamma1, gamma2, gamma3, gamma4]
block13_in_var = [var1, var2, var3, var4]
block13_in_mean = [mean1, mean2, mean3, mean4]

#block 14
i = 14
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block14_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block14_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block14_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block14_in_var = [var1, var2, var3, var4, var5, var6]
block14_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]


#block 15
i = 15
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block15_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block15_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block15_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block15_in_var = [var1, var2, var3, var4, var5, var6]
block15_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]

#block 16
i = 16
conv1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d/kernel"]
conv2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_1/kernel"]
conv3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/conv2d_2/kernel"]
conv4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d/kernel"]
conv5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_1/kernel"]
conv6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/conv2d_2/kernel"]

beta1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/beta"]
beta2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/beta"]
beta3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/beta"]
beta4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/beta"]
beta5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/beta"]
beta6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/beta"]

gamma1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/gamma"]
gamma2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/gamma"]
gamma3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/gamma"]
gamma4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/gamma"]
gamma5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/gamma"]
gamma6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/gamma"]

var1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_variance"]
var2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_variance"]
var3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_variance"]
var4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_variance"]
var5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_variance"]
var6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_variance"]

mean1 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization/moving_mean"]
mean2 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_1/moving_mean"]
mean3 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_0/batch_normalization_2/moving_mean"]
mean4 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization/moving_mean"]
mean5 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_1/moving_mean"]
mean6 = weights[f"spinenet/sub_policy{i-2}/resample_with_alpha_resample_{i-2}_1/batch_normalization_2/moving_mean"]

block16_in_conv = [conv1, conv2, conv3, conv4, conv5, conv6]
block16_in_beta = [beta1, beta2, beta3, beta4, beta5, beta6]
block16_in_gamma = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
block16_in_var = [var1, var2, var3, var4, var5, var6]
block16_in_mean = [mean1, mean2, mean3, mean4, mean5, mean6]