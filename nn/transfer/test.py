import keras
from keras import Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam

INPUT_WIDTH = int(4 * 144 / 3)
INPUT_HEIGHT = 30 * 4

densenet_1 = keras.applications.densenet.DenseNet121(include_top=False, 
    weights='imagenet', 
    input_shape=(INPUT_WIDTH, INPUT_HEIGHT, 3))
densenet_1.name = "densenet_1"
for layer in densenet_1.layers:
    layer.name += "_1"
    layer.trainable = False
#input1 = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
#densenet_1.layers[0] = input1
x1 = densenet_1
x1 = Flatten()(x1.output)

densenet_2 = keras.applications.densenet.DenseNet121(include_top=False, 
    weights='imagenet', 
    input_shape=(INPUT_WIDTH, INPUT_HEIGHT, 3))
densenet_2.name = "densenet_2"
for layer in densenet_2.layers:
    layer.name += "_2"
    layer.trainable = False
#input2 = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
#densenet_2.layers[0] = input2
x2 = densenet_2
x2 = Flatten()(x2.output)

x = keras.layers.concatenate([x1.output, x2.output])
x = Dense(2000)(x)
x = Dense(2000)(x)
x = Dense(1440)(x)
x = Reshape((30, 48, 1))(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

model = Model(inputs=[densenet_1.input, densenet_2.input], outputs=[x])
adam = Adam(lr=0.000025)
model.compile(adam, loss='binary_crossentropy')
model.summary()

