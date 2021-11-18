# import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.utils import conv_utils
import numpy as np
from tensorflow.python.keras import layers, models, activations, initializers, constraints
from math import cos, pi, sqrt
from tensorflow.python.keras.regularizers import l2

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Space2Depth(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(Space2Depth, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return tf.nn.space_to_depth(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, int(input_shape[1]/self.scale), int(input_shape[2]/self.scale), input_shape[3]*self.scale**2)
        else:
            return (None, None, None, input_shape[3]*self.scale**2)

class Depth2Space(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(Depth2Space, self).__init__(**kwargs)
        self.scale = scale
    def call(self, inputs, **kwargs):
        return tf.depth_to_space(inputs, self.scale)

        # return tf.disable_v2_behavior(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, input_shape[1]*self.scale, input_shape[2]*self.scale, int(input_shape[3]/self.scale**2))
        else:
            return (None, None, None, int(input_shape[3]/self.scale**2))


class adaptive_implicit_trans(layers.Layer):
    def __init__(self, **kwargs):
        super(adaptive_implicit_trans, self).__init__(**kwargs)

    def build(self, input_shape):
        conv_shape = (1,1,64,64)
        self.it_weights = self.add_weight(
            shape = (1,1,64,1),
            initializer = initializers.get('ones'),
            constraint = constraints.NonNeg(),
            name = 'ait_conv')
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        sess = tf.InteractiveSession()
        self.kernel = k.variable(value = kernel, dtype = 'float32')


        # with sess.as_default():
        #     print(self.kernel.eval())
        # exit()
        # print(self.it_weights)

    def call(self, inputs):
        #it_weights = k.softmax(self.it_weights)
        #self.kernel = self.kernel*it_weights
        # print('self.kernel    \t\t',self.kernel) # <tf.Variable 'adaptive_implicit_trans_1/Variable:0' shape=(1, 1, 64, 64) dtype=float32>

        # print('self.it_weights\t\t',self.it_weights) #<tf.Variable 'adaptive_implicit_trans_1/ait_conv:0' shape=(1, 1, 64, 1) dtype=float32>
        self.kernel = self.kernel #* self.it_weights

        y = k.conv2d(inputs,
                        self.kernel,
                        padding = 'same',
                        data_format='channels_last')

        # print('self.kernel    \t\t',self.kernel) # Tensor("adaptive_implicit_trans_1/mul:0", shape=(1, 1, 64, 64)
        # print('inputs',inputs)
        # print(' y',y)


        # print('type(self.kernel)',self.kernel)
        # print('type(self.it_weights)',self.it_weights)
        # print('kernel.shape',self.kernel.shape)
        # print('it_weights.shape',self.it_weights.shape)
        # print('y =  \t',y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaleLayer(layers.Layer):
    def __init__(self, s, **kwargs):
        self.s = s
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # print('input_shape = \t',input_shape)
        self.kernel = self.add_weight(
            shape = (1,),
            name = 'scale',
            initializer=initializers.Constant(value=self.s))

    def call(self, inputs):
        # print('input',input)
        # print('kernel',self.kernel)
        # print(self.kernel)
        return inputs*self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
