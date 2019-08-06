import h5py
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import os
import time
from PIL import Image
import colorsys

INPUT_WIDTH = int(144 / 3)
INPUT_HEIGHT = 30 * 2
OUTPUT_WIDTH = int(144 / 3)
OUTPUT_HEIGHT = 30

X_MUL = 1.0 / 255.0
Y_MUL = 1.0 / 100.0

def save_rgb_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = np.max(result)

    img = Image.new('RGBA', (width, height))

    for i in range(width):
        for j in range(height):
            values = result[j][i];
            if isinstance(values, list):
                (r,g,b) = (int(values[0]), int(values[1]), int(values[2]))
            else:
                (r,g,b) = (int(values), int(values), int(values))
            img.putpixel((i, j), (r, g, b, 255))

    img.save(image_path, "PNG")

def save_depth_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = np.max(result)

    img = Image.new('RGBA', (width, height))

    for i in range(width):
        for j in range(height):
            if result[j][i] == -1:
                continue;

            h = 240 * result[j][i] / maxVal;
            (r,g,b) = colorsys.hsv_to_rgb(h / 360,1,1)

            img.putpixel((i, j), (int(r * 255), int(g * 255), int(b * 255), 255))

    img.save(image_path, "PNG")

def concat_left_right_datasets(d1, d2):
    if len(d1) != len(d2):
        print("Datasets are not the same size.")
        exit()

    d = []

    for i in range(len(d1)):
        r1 = d1[i]
        r2 = d2[i]

        if len(r1) != len(r2):
            print("Dataset item #", i, "in two sets does not have the same INPUT_HEIGHT")
            exit()

        if len(r1[0]) != len(r2[0]):
            print("Dataset item #", i, "in two sets does not have the same INPUT_WIDTH")
            exit()

        r = []

        for j in d1[i]:
            r.append(j)
        for j in d2[i]:
            r.append(j)
        
        d.append(r)

    return d

def load_h5(path):
    f = h5py.File(path, 'r')
    a_group_key = list(f.keys())[0]

    print('Loading dataset', a_group_key, '...')
    data = list(f[a_group_key])

    return data;


depth_train_02 = load_h5('../../create_h5_dataset/depth_train_02_0.h5')
#depth_train_03 = load_h5('../../create_h5_dataset/depth_train_03_0.h5')
rgb_train_02 = load_h5('../../create_h5_dataset/rgb_train_02_0.h5')
rgb_train_03 = load_h5('../../create_h5_dataset/rgb_train_03_0.h5')
rgb_train = concat_left_right_datasets(rgb_train_02, rgb_train_03)

depth_eval_02 = load_h5('../../create_h5_dataset/depth_eval_02_0.h5')
#depth_train_03 = load_h5('../../create_h5_dataset/depth_train_03_0.h5')
rgb_eval_02 = load_h5('../../create_h5_dataset/rgb_eval_02_0.h5')
rgb_eval_03 = load_h5('../../create_h5_dataset/rgb_eval_03_0.h5')
rgb_eval = concat_left_right_datasets(rgb_eval_02, rgb_eval_03)




input_img = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1))
layer = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
layer = MaxPooling2D((2,2), padding='same')(layer)
layer = Conv2D(16, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((2,2), padding='same')(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((2,2), padding='same')(layer)

layer = Flatten()(layer)
layer = Dense(384, activation='relu')(layer)
layer = Dense(360, activation='relu')(layer)

layer = Reshape((15, 24, 1))(layer)

layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = UpSampling2D((2, 2))(layer)
layer = Conv2D(1, (3, 3), activation='relu', padding='same')(layer)


model = Model(input_img, layer)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()

x_train = np.reshape(np.array(rgb_train) * X_MUL, (len(rgb_train), INPUT_HEIGHT, INPUT_WIDTH, 1))
y_train = np.reshape(np.array(depth_train_02) * Y_MUL, (len(depth_train_02), OUTPUT_HEIGHT, OUTPUT_WIDTH, 1))
x_eval = np.reshape(np.array(rgb_eval) * X_MUL, (len(rgb_eval), INPUT_HEIGHT, INPUT_WIDTH, 1))
y_eval = np.reshape(np.array(depth_eval_02) * Y_MUL, (len(depth_eval_02), OUTPUT_HEIGHT, OUTPUT_WIDTH, 1))

test_sample_x = x_eval[0]
test_sample_y = y_eval[0]

logdir = 'log/' + time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists('log'):
    os.mkdir('log')
if not os.path.exists(logdir):
    os.mkdir(logdir)

save_rgb_image(test_sample_x / X_MUL, logdir + '/_x.png')
save_depth_image(test_sample_y / Y_MUL, logdir + '/_y.png')

for i in range(10000):
    print("Epoch", i)
    model.fit(x_train, y_train, 
        epochs=1,
        batch_size=256,
        shuffle=True,
        validation_data=(x_eval, y_eval))

    yy = model.predict(np.array([test_sample_x]))
    print(yy.shape)    
    save_depth_image(yy[0] / Y_MUL, logdir + '/' + str(i) + '.png')
    
