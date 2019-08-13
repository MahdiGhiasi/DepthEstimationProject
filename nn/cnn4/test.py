
INPUT_WIDTH = int(4 * 144 / 3)
INPUT_HEIGHT = 30 * 4
OUTPUT_WIDTH = int(4 * 144 / 3)
OUTPUT_HEIGHT = 30 * 4

import keras
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D, ReLU, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam


# Create a new model

# Feature extractor
l1 = Conv2D(64, (5, 5), activation='relu', padding='same')
l2 = MaxPooling2D((2,2), padding='same')
l3 = Conv2D(32, (3, 3), activation='relu', padding='same')
l4 = MaxPooling2D((2,2), padding='same')
l5 = Conv2D(16, (3, 3), activation='relu', padding='same')
l6 = MaxPooling2D((2,2), padding='same')
l7 = Conv2D(8, (3, 3), activation='relu', padding='same')

# Models for two inputs use the exact same feature extractor network with the same weights
input_img_1 = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
x_1 = l1(input_img_1)
x_1 = l2(x_1)
skip_1 = MaxPooling2D((4, 4), padding='same')(x_1)
x_1 = l3(x_1)
x_1 = l4(x_1)
x_1 = l5(x_1)
x_1 = l6(x_1)
x_1 = l7(x_1)
x_1 = Flatten()(x_1)

input_img_2 = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
x_2 = l1(input_img_2)
x_2 = l2(x_2)
x_2 = l3(x_2)
x_2 = l4(x_2)
x_2 = l5(x_2)
x_2 = l6(x_2)
x_2 = l7(x_2)
x_2 = Flatten()(x_2)

x = Concatenate()([x_1, x_2])
#layer = Dense(2000, activation='relu')(layer)
#layer = Dense(1440, activation='relu')(layer)
x = Dense(500, activation='relu')(x)
x = Dense(360, activation='relu')(x)
x = Dense(360, activation='relu')(x)

x = Reshape((15, 24, 1))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = keras.layers.concatenate([x, skip_1])
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

model = Model([input_img_1, input_img_2], x)

adam = Adam(lr=0.000025)
model.compile(adam, loss='binary_crossentropy')
model.summary()