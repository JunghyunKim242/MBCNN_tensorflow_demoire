# from keras import layers

from tensorflow.python.keras import layers
# from keras.models import Model
# from keras import models
from tensorflow.keras import models
from tensorflow.python.keras import backend as K
from core_layers import *


def conv_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = layers.Conv2D(filters,1,padding=padding,use_bias=use_bias,
            activation='relu')(x)
    else:
        y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            activation='relu')(x)
    return y


def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    return y


def conv_bn_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate)(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    return y


def conv_prelu(x, filters, kernel, padding='same', use_bias=False, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    y = layers.advanced_activations.PReLU()(y)
    return y


def MBCNN(nFilters, multi=True):
    print("this is model")

    conv_func = conv_relu
    def pre_block(x, d_list, enbale = True):
        t = x
        for i in range(len(d_list)): # 논문에서는 5번
            # print('\ni =',i)
            # print('before pre_block = ', t.shape)
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            # print('after pre_block = ', _t.shape)
            t = layers.Concatenate(axis=-1)([_t,t])
            # print('pre_block = ', t.shape)


        # print('conv1 = ', t.shape)  #512,512,448
        t = conv(t, 64, 3)
        print('\nAdaptive_implicit_trans_conv1 = ', t.shape)  #512,512,64
        t = adaptive_implicit_trans()(t)
        print('Adaptive_implicit_trans_conv12 = = ', t.shape)  #512,512,64




        t = conv(t,nFilters*2,1)
        # print('conv1 = ', t.shape) #512,512,128
        t = ScaleLayer(s=0.1)(t)
        # print('conv1 = ', t.shape) #512,512,128

        if not enbale:
            t = layers.Lambda(lambda x: x*0)(t)

        # print('Add = ', t.shape)  # 128, 128, 192
        t = layers.Add()([x,t])
        # print('Add = ', t.shape)  # 128, 128, 192
        return t


    def pos_block(x, d_list):
        t = x
        for i in range(len(d_list)):
            # print('\npos_block1 = ',t.shape) #  128, 128, 128)
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            # print('pos_block2 = ',_t.shape) #  128, 128, 64
            # print('pos_block3 = ',t.shape) #  128, 128, 128)
            t = layers.Concatenate(axis=-1)([_t,t])
            # print('pos_block4 = ',t.shape)  #  128, 128, 192

        # print('pos_blocklast = ', t.shape)  # 128, 128, 192
        t = conv_func(t, nFilters*2, 1)
        # print('pos_blocklast = ', t.shape)  # 128, 128, 192

        return t


    def global_block(x):
        # print('\n BEFORE zeropadding = ', x.shape)  #128,128,128,    258,258,128
        t = layers.ZeroPadding2D(padding=(1,1))(x)
        # print('global_blocktsize1 = ',t.shape)  #130,130,128,    258,258,128
        t = conv_func(t, nFilters*4, 3, strides=(2,2))
        # print('input of Global_AveragePooling 2D  = ', t.shape)  #65, 65, 256     129,129,256
        t = layers.GlobalAveragePooling2D()(t)
        # print('after global Global_AveragePooling 2D \t', t.shape)    #256
        t = layers.Dense(nFilters*16, activation='relu')(t)
        # print('After Dense 1\t', t.shape)    #1024
        t = layers.Dense(nFilters*8, activation='relu')(t)
        # print('After Dense 2\t', t.shape)    #512
        t = layers.Dense(nFilters*4)(t)
        # print('After Dense 3\t', t.shape)    #256

        # print('input of conv function = ', x.shape)  #128,128,128   512,512,128
        _t = conv_func(x, nFilters*4, 1)
        # print('after conv_func = ', _t.shape) # 128,128,256,    512,512,256


        # print('input of mul = ',_t.shape)   # 128,128,256
        # print('input of mul = ',t.shape)    # ?,256
        _t = layers.Multiply()([_t,t])
        # print('input of conv function second = ', _t.shape) #128,128,256    512,512,256
        _t = conv_func(_t, nFilters*2, 1)
        # print('after conv_func second = ', _t.shape) # 128,128,128  512,512,128
        return _t

    size = 1024
    output_list = []
    d_list_a = (1,2,3,2,1)
    d_list_b = (1,2,3,2,1)
    d_list_c = (1,2,2,2,1)

    x = layers.Input(shape=(size, size, 3))                 #16m*16m
    # print('x.shape',x.shape) # 1024, 1024, 3
    # print('type(x)',type(x))    #<class 'tensorflow.python.framework.ops.Tensor'>
    _x = Space2Depth(scale=2)(x)


    # print('140_x.shape',_x.shape) #512, 512, 12
    t1 = conv_func(_x, nFilters*2, 3, padding='same')          #8m*8m # 512,512,128
    #print('142t1.shape',t1.shape) # 512,512,128
    t1 = pre_block(t1, d_list_a, True)
    #print('144t1.shape',t1.shape) # 512,512,128
    t2 = layers.ZeroPadding2D(padding=(1,1))(t1)


    #print('\nFlag1 t2.shape',t2.shape) #514,514,128
    t2 = conv_func(t2, nFilters*2, 3, padding='valid',strides=(2,2))              #4m*4m
    #print('144t2.shape',t2.shape) # 256,256,128
    t2 = pre_block(t2, d_list_b,True)
    #print('layers.146 t2.shape',t2.shape)# 256,256,128
    t3 = layers.ZeroPadding2D(padding=(1,1))(t2)


    #print('\nFlag2 t3.shape',t3.shape) #258,258,128
    t3 = conv_func(t3, nFilters*2, 3, padding='valid',strides=(2,2))              #2m*2m
    #print('t3.shape',t3.shape) # 128,128,128,
    t3 = pre_block(t3, d_list_c, True)
    #print('154 global block input t3.shape',t3.shape) # 128,128,128,
    t3 = global_block(t3)
    #print('156t3.shape',t3.shape) # 128,128,128,
    t3 = pos_block(t3, d_list_c)
    #print('158t3.shape',t3.shape) # 128,128,128,
    t3_out = conv(t3, 12, 3)
    #print('\nt3_out.shape',t3_out.shape) # 128,128,12,
    t3_out = Depth2Space(scale=2)(t3_out)           #4m*4m
    #print('t3_out.shape',t3_out.shape) # 256,256,3
    # t3_out = tf.disable_v2_behavior(scale=2)(t3_out)           #4m*4m
    output_list.append(t3_out)
    #print('\n\noutput_list.shape',len(output_list)) # 1


    _t2 = layers.Concatenate()([t3_out,t2])
    #print('_t2.shape',_t2.shape) #  256,256,131
    _t2 = conv_func(_t2, nFilters*2, 1)
    #print('_t2.shape',_t2.shape) # 256,256,128
    _t2 = global_block(_t2)
    #print('_t2.shape',_t2.shape) # 256,256,128
    _t2 = pre_block(_t2, d_list_b,True)
    #print('_t2.shape',_t2.shape) # 256,256,128
    _t2 = global_block(_t2)
    #print('_t2.shape',_t2.shape) # 256,256,128
    _t2 = pos_block(_t2, d_list_b)
    #print('_t2.shape', _t2.shape) # 256,256,128
    t2_out = conv(_t2, 12, 3)
    #print('\nt2_out.shape', t2_out.shape) # 256,256,12
    t2_out = Depth2Space(scale=2)(t2_out)           #8m*8m
    #print('t2_out.shape', t2_out.shape) # 512,512,3
    # t2_out = tf.disable_v2_behavior(scale=2)(t2_out)           #8m*8m
    output_list.append(t2_out)
    #print('\n\nlen(output_list)',len(output_list)) # 2


    _t1 = layers.Concatenate()([t1, t2_out])
    #print('\noutput torch.cat ', _t1.shape)  # 4,512,512,131
    _t1 = conv_func(_t1, nFilters*2, 1)
    #print('\noutput conv_func5  ',_t1.shape)    #4,512,512,128
    _t1 = global_block(_t1)
    #print('\noutput global_block4 ',_t1.shape)  #,512,512,128
    _t1 = pre_block(_t1, d_list_a, True)
    #print('\noutput pre_block5  ',_t1.shape)
    _t1 = global_block(_t1)
    #print('\noutput global_block5 ', _t1.shape)
    _t1 = pos_block(_t1, d_list_a)
    #print('\noutput global_block5 ', _t1.shape)
    _t1 = conv(_t1,12,3)
    #print('\noutput conv3 ', _t1.shape)
    y = Depth2Space(scale=2)(_t1)


    #print(' \ny.shape', y.shape) # 1024,1024,12                  #16m*16m
    # y = tf.disable_v2_behavior(scale=2)(_t1)                           #16m*16m
    output_list.append(y)
    #print(' \nlen(outputlist)',len(output_list) )  #3                #16m*16m

    if multi != True:
        return models.Model(x,y)
    else:
        return models.Model(x,output_list)
