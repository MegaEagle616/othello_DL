from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Input,
    Reshape,
    GlobalAveragePooling2D,
    Dense,
    Conv2D,
    BatchNormalization,
    Activation,
    add,
    ReLU,
    Concatenate,
    Dropout,
    AveragePooling2D,
    MaxPooling2D
)
from keras.optimizers import adam_v2
from keras.regularizers import l2


import numpy as np, os


class OthelloModel:
    def __init__(self, input_shape=(8, 8)):
        self.model_name = 'model_'+('x'.join(list(map(str, input_shape))))+'.h5'
        self.input_boards = Input(shape=input_shape)
        x_image = Reshape(input_shape+(1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs=x_image, num_res_blocks=2)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        self.pi = Dense(input_shape[0]*input_shape[1], activation='softmax', name='pi')(gap1)
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(loss=['categorical_crossentropy'], optimizer=adam_v2.Adam(0.002))
        
    def predict(self, board):
        return self.model.predict(np.array([board]).astype('float32'))[0]
    
    def fit(self, data, batch_size, epochs):
        input_boards, target_policys = list(zip(*data))
        input_boards = np.array(input_boards)
        target_policys = np.array(target_policys)
        self.model.fit(x=input_boards, y=[target_policys], batch_size = batch_size, epochs=epochs)
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def save_weights(self):
        self.model.save_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def load_weights(self):
        self.model.load_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def reset(self, confirm=False):
        if not confirm:
            raise Exception('this operate would clear model weight, pass confirm=True if really sure')
        else:
            try:
                os.remove('othello/bots/DeepLearning/models/'+self.model_name)
            except:
                pass
        print('cleared')

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for  i in range(1):
            resnet = self.resnet_layer(inputs=x, num_filter=128)
            resnet = self.resnet_layer(inputs=resnet, num_filter=128, activation=None)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if i == 0:
                resnet = self.resnet_layer(inputs= x, num_filter=256, strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter=256, activation=None)
            else:
                resnet = self.resnet_layer(inputs=x, num_filter=256)
                resnet = self.resnet_layer(inputs=resnet, num_filter=256, activation=None)
            if i == 0:
                x = self.resnet_layer(inputs=x, num_filter=256, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if i == 0:
                resnet = self.resnet_layer(inputs = x, num_filter = 512,strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation=None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 512)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation=None)
            if i == 0:
                x = self.resnet_layer(inputs=x, num_filter = 512, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet
        return x

    def resnet_layer(self, inputs, num_filter=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, padding='same'):
        
        conv = Conv2D(num_filter, 
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=False,
                    kernel_regularizer=l2(1e-4))
        
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            
        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def DenseLayer(self, growthRate, dropRate=0.2):

        # bottleneck
        x = BatchNormalization(axis=3)(self)
        x = ReLU(x)
        x = Conv2D(4 * growthRate, kernel_size=(1, 1), padding='same')(x)

        # composition
        x = BatchNormalization(axis=3)(x)
        x = ReLU(x)
        x = Conv2D(growthRate, kernel_size=(3, 3), padding='same')(x)

        # dropout
        x = Dropout(dropRate)(x)

        return x

    def DenseBlock(self, num_layers, growthRate, dropRate=0):

        for i in range(num_layers):
            featureMap = self.DenseLayer(x, growthRate, dropRate)
            x = Concatenate([x, featureMap], axis=3)

        return x

    def TransitionLayer(self, ratio):

        growthRate = int(self.shape[-1] * ratio)
        x = BatchNormalization(axis=3)(self)
        x = ReLU(x)
        x = Conv2D(growthRate, kernel_size=(1, 1),
                   strides=(2, 2), padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        return x

    def DenseNet121(self, numClass=1000, inputShape=(224, 224, 3), growthRate=12):
        x_in = Input(inputShape)
        x = Conv2D(growthRate * 2, (3, 3), padding='same')(x_in)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self.TransitionLayer(self.DenseBlock(
            x, num_layers=6, growthRate=12, dropRate=0.2))
        x = self.TransitionLayer(self.DenseBlock(
            x, num_layers=12, growthRate=12, dropRate=0.2))
        x = self.TransitionLayer(self.DenseBlock(
            x, num_layers=24, growthRate=12, dropRate=0.2))

        x = self.DenseBlock(x, num_layers=16, growthRate=12, dropRate=0.2)
        x = GlobalAveragePooling2D()(x)
        x_out = Dense(numClass=1000, activation='softmax')(x)

        model = Model(x_in, x_out)
        model.compile(optimizer=adam_v2.Adam(),
                      loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.summary()
        return model

        
