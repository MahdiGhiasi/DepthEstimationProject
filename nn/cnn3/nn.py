import h5py
import numpy as np
import os
import time
from PIL import Image
import colorsys
import random
import gc
import sys
from multiprocessing import Process

INPUT_WIDTH = int(4 * 144 / 3)
INPUT_HEIGHT = 30 * 4 * 2
OUTPUT_WIDTH = int(4 * 144 / 3)
OUTPUT_HEIGHT = 30 * 4

X_MUL = 1.0 / 255.0
Y_MUL = 1.0 / 100.0

TRAIN_DATASET_MAX_INDEX = 20
TRAIN_DATASET_ITEM_LOAD_EACH_TIME = 2
TRAIN_EPOCH_COUNT_ON_EACH_SUBSET = 20

def save_rgb_image(result, image_path):
    height = len(result)
    width = len(result[0])

    img = Image.new('RGBA', (width, height))

    for i in range(width):
        for j in range(height):
            values = result[j][i];
            if isinstance(values, list) or isinstance(values, np.ndarray):
                (r,g,b) = (int(values[0]), int(values[1]), int(values[2]))
            else:
                (r,g,b) = (int(values), int(values), int(values))
            img.putpixel((i, j), (r, g, b, 255))

    img.save(image_path, "PNG")

def save_depth_image(result, image_path):
    height = len(result)
    width = len(result[0])
    
    maxVal = 100

    img = Image.new('RGBA', (width, height))

    for i in range(width):
        for j in range(height):
            if result[j][i] == -1:
                continue;

            h = 240 * min(result[j][i], maxVal) / maxVal;
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


def load_dataset_subset():
    depth_train_02 = []
    rgb_train_02 = []
    rgb_train_03 = []
    
    indexes = []
    curIndex = random.randint(0, TRAIN_DATASET_MAX_INDEX)
    while len(indexes) < TRAIN_DATASET_ITEM_LOAD_EACH_TIME:
        if curIndex in indexes:
            continue
        indexes.append(curIndex)
        curIndex = random.randint(0, TRAIN_DATASET_MAX_INDEX)

    for i in indexes:
        print('loading dataset subset', i, '...')
        depth_train_02 += load_h5('i:/dataset_120/depth_train_02_' + str(i) + '.h5')
        #depth_train_03 += load_h5('../../create_h5_dataset/depth_train_03_0.h5')
        rgb_train_02 += load_h5('i:/dataset_120/rgb_train_02_' + str(i) + '.h5')
        rgb_train_03 += load_h5('i:/dataset_120/rgb_train_03_' + str(i) + '.h5')

    rgb_train = concat_left_right_datasets(rgb_train_02, rgb_train_03)

    del rgb_train_02
    del rgb_train_03
    
    gc.collect()

    return (depth_train_02, rgb_train)


def train_model(model_input_file, model_output_file, x_eval, y_eval, test_sample_x, test_sample_index, logdir):
    (depth_train, rgb_train) = load_dataset_subset()

    x_train = np.reshape(np.array(rgb_train) * X_MUL, (len(rgb_train), INPUT_HEIGHT, INPUT_WIDTH, 3))
    y_train = np.reshape(np.array(depth_train) * Y_MUL, (len(depth_train), OUTPUT_HEIGHT, OUTPUT_WIDTH, 1))

    from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model, load_model
    from keras.optimizers import Adam

    if len(model_input_file) == 0:
        # Create a new model
        input_img = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
        layer = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
        layer = MaxPooling2D((2,2), padding='same')(layer)
        layer = Conv2D(32, (3, 3), activation='relu', padding='same')(layer)
        layer = MaxPooling2D((2,2), padding='same')(layer)
        layer = Conv2D(16, (3, 3), activation='relu', padding='same')(layer)
        layer = MaxPooling2D((2,2), padding='same')(layer)
        layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
        #layer = MaxPooling2D((2,2), padding='same')(layer)

        layer = Flatten()(layer)
        #layer = Dense(2000, activation='relu')(layer)
        #layer = Dense(1440, activation='relu')(layer)
        layer = Dense(1800, activation='relu')(layer)
        layer = Dense(1440, activation='relu')(layer)
        layer = Dense(1440, activation='relu')(layer)

        layer = Reshape((30, 48, 1))(layer)

        #layer = Conv2D(1, (3, 3), activation='relu', padding='same')(layer)
        layer = UpSampling2D((2, 2))(layer)
        #layer = Conv2D(1, (3, 3), activation='relu', padding='same')(layer)
        layer = UpSampling2D((2, 2))(layer)
        #layer = Conv2D(1, (3, 3), activation='relu', padding='same')(layer)

        model = Model(input_img, layer)

        adam = Adam(lr=0.000025)
        model.compile(adam, loss='binary_crossentropy')
        model.summary()

    else:
        model = load_model(model_input_file)

    #model.optimizer.lr.set_value(0.00003)

    model.fit(x_train, y_train, 
        epochs=TRAIN_EPOCH_COUNT_ON_EACH_SUBSET,
        batch_size=12,
        shuffle=True,
        validation_data=(x_eval, y_eval))

    yy = model.predict(np.array(test_sample_x))

    model.save(model_output_file)

    for i in range(len(yy)):
        save_depth_image(yy[i] / Y_MUL, logdir + '/' + str(i) + '/' + str(test_sample_index) + '.png')
    

if __name__ == '__main__':

    depth_eval_02 = load_h5('i:/dataset_120/depth_eval_02_0.h5')
    #depth_train_03 = load_h5('../../create_h5_dataset/depth_train_03_0.h5')
    rgb_eval_02 = load_h5('i:/dataset_120/rgb_eval_02_0.h5')
    rgb_eval_03 = load_h5('i:/dataset_120/rgb_eval_03_0.h5')
    rgb_eval = concat_left_right_datasets(rgb_eval_02, rgb_eval_03)

    x_eval = np.reshape(np.array(rgb_eval) * X_MUL, (len(rgb_eval), INPUT_HEIGHT, INPUT_WIDTH, 3))
    y_eval = np.reshape(np.array(depth_eval_02) * Y_MUL, (len(depth_eval_02), OUTPUT_HEIGHT, OUTPUT_WIDTH, 1))

    test_ctr = 0
    test_sample_x = []
    test_sample_y = []
    while test_ctr < len(x_eval):
        test_sample_x.append(x_eval[test_ctr])
        test_sample_y.append(y_eval[test_ctr])
        test_ctr += 25

    logdir = 'log/' + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    for i in range(len(test_sample_x)):
        if not os.path.exists(logdir + '/' + str(i)):
            os.mkdir(logdir + '/' + str(i))

    for i in range(len(test_sample_x)):
        save_rgb_image(test_sample_x[i] / X_MUL, logdir + '/' + str(i) + '/_x.png')
        save_depth_image(test_sample_y[i] / Y_MUL, logdir + '/' + str(i) + '/_y.png')

    counter = 0

    output_file = ""
    input_file = ""
    old_model = ""

    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        print("Loading NN from ", output_file)

    while True:
        print()
        print("Epoch", counter)
            
        counter += TRAIN_EPOCH_COUNT_ON_EACH_SUBSET
        
        old_model = input_file
        input_file = output_file
        output_file = logdir + '/model-epoch' + str(counter) + '.h5'

        if len(old_model) > 0:
            os.remove(old_model)

        # GPU Crashes after calling fit() multiple times, so we call it in a separate process every time,
        # and save the model and exit Keras.
        # https://stackoverflow.com/questions/47118723/cntk-out-of-memory-error-when-model-fit-is-called-second-time
        process = Process(target=train_model, args=[input_file, output_file, x_eval, y_eval, test_sample_x, counter, logdir])
        process.start()
        process.join()
