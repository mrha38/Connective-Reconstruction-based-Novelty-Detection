import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tools import charbonnier_loss, mixData


class ConnectiveModel:

    def __init__(self):
        self.latentSpaceSize = 256
        self.encoder = self.buildEncoder()
        self.decoder, self.AE = self.buildAE(self.encoder)
        self.classifier, self.connective = self.buildNoveltyModel()


    def buildEncoder(self):
        inputs = layers.Input(shape=(32,32,1))
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        previous_block_activation = x
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(0.3)(x)
        residual = layers.Conv2D(64, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual]) 
        previous_block_activation = x
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(0.3)(x)
        residual = layers.Conv2D(128, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(0.5)(x)
        residual = layers.Conv2D(256, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x
        latentSpace = layers.Flatten(name='flatten')(previous_block_activation)
        model_1_output = layers.Dense(self.latentSpaceSize, activation='relu')(latentSpace)
        model = Model(inputs, model_1_output)
        return model


    def buildAE(self, encoderModel):  
        # encoderModel.trainable = False
        inputs = layers.Input(shape=(32,32,1))
        inputLatentSpace = layers.Input(shape=(self.latentSpaceSize,))
        
        x2 = layers.Dense(1024, name = 'Dense6', activation = 'relu')(inputLatentSpace)
        x2 = layers.Reshape((2, 2, 256))(x2)
        x2 = layers.Conv2DTranspose(256, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(256, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.UpSampling2D(2)(x2)
        x2 = layers.Dropout(0.3)(x2)
        previous_block_activation = x2 

        # # 128
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(128, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(128, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.UpSampling2D(2)(x2)
        
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(128, 1, padding="same")(residual)
        residual = layers.Activation("relu")(residual)
        x2 = layers.add([x2, residual]) 
        x2 = layers.Dropout(0.3)(x2)
        previous_block_activation = x2

        # 64
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(64, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(64, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.UpSampling2D(2)(x2)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(64, 1, padding="same")(residual)
        residual = layers.Activation("relu")(residual)
        x2 = layers.add([x2, residual])
        x2 = layers.Dropout(0.3)(x2)
        previous_block_activation = x2
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(32, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.Conv2DTranspose(32, 3, padding="same")(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation("relu")(x2)
        x2 = layers.UpSampling2D(2)(x2)
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(32, 1, padding="same")(residual)
        residual = layers.Activation("relu")(residual)
        x2 = layers.add([x2, residual]) 
        x2 = layers.Dropout(0.3)(x2)
        previous_block_activation = x2 
        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x2)
        baseDecoder = Model(inputLatentSpace, outputs)
        encoderModelOutPut = encoderModel(inputs)
        baseDecoderOutput = baseDecoder(encoderModelOutPut)
        model = Model(inputs, baseDecoderOutput)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer, loss=charbonnier_loss
        )
        return baseDecoder, model


    def buildNoveltyModel(self):
        self.AE.trainbale = False
        n1_input = layers.Input(shape=(32,32,1))
        ae_input = layers.Input(shape=(32,32,1))
        n1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(n1_input)
        n1 = layers.MaxPooling2D(pool_size=(2, 2))(n1)
        n1 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(n1)
        n1 = layers.MaxPooling2D(pool_size=(2, 2))(n1)
        n1 = layers.Flatten()(n1)
        n1 = layers.Dropout(0.5)(n1)
        n1 = layers.Dense(256, activation="relu")(n1)
        n1 = layers.Dense(128, activation="relu")(n1)
        n1 = layers.Dense(2, activation="softmax")(n1)

        model = Model(n1_input, n1)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        aeOT = self.AE(ae_input)
        mixLayer = layers.Lambda(mixData)([ae_input, aeOT])
        baseModelOT = model(mixLayer)

        noveltyModel = Model(ae_input, baseModelOT)
        noveltyModel.compile(optimizer='adam', loss='binary_crossentropy')

        return model, noveltyModel


    