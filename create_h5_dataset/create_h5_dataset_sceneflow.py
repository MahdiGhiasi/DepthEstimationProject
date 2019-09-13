import os
import json
from threading import Thread
import h5py
import random
import gzip
from multiprocessing import Process

WIDTH = 192
HEIGHT = 120

CHUNK_MAX_SIZE = 1100

CROP_SETTING = 'none' # 'crop3' or 'none'

GZIP = True

# THREAD_COUNT = 1

# def load_data(files_keyvaluepair, result, index):
#     data = {}
#     for key, value in files_keyvaluepair:
#         with open(value, 'r') as file:
#             content_string = file.read()

#         content = jsonpickle.decode(content_string)
#         data[key] = content

#         if len(data) % 100 == 0:
#             print(len(data), "files loaded")

#     result[index] = data

def crop_w(image, start, length):
    output = []
    for row in image:
        output.append(row[start:(start+length)])

    return output


def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# def load_data_parallel(files):
#     files_keyvaluepair = [[k, v] for k, v in files.items()]

#     files_keyvaluepair_splitted = chunk(files_keyvaluepair, THREAD_COUNT)

#     threads = [None] * THREAD_COUNT
#     results = [None] * THREAD_COUNT

#     for i in range(THREAD_COUNT):
#         threads[i] = Thread(target=load_data, args=(files_keyvaluepair_splitted[i], results, i))
#         threads[i].start()


#     for i in range(THREAD_COUNT):
#         threads[i].join()

#     print(len(results[0]))
#     print(len(results[1]))

def load_files(input_path):
    data = {}
    found_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".json.gz" if GZIP else ".json"):
                full_path = os.path.join(root, file)
                found_files.append(full_path)

    for full_path in found_files:
        path_chunks = full_path.split(os.path.sep)

        key = ""
        for s in path_chunks:
            if s.endswith("_x2"):
                key += s + "__"
            elif s == "left" or s == "right":
                key += s + "__"
            elif s.isnumeric():
                key += s + "__"

        key += path_chunks[-1][1:]
        
        data[key] = full_path

        #with open(full_path, 'r') as file:
        #    content_string = file.read()
        
        #print(key)

        #content = jsonpickle.decode(content_string)
        #print(data)
        #return

        #if len(data) % 100 == 0:
        #    print(len(data), "/", len(found_files), "loaded")

    return data

def create_h5(keys, files, name, output):
    dataset = []

    total_count = len(keys)
    if CROP_SETTING == 'crop3':
        total_count *= 3

    print("creating dataset", name, "...")
    for key in keys:
        full_path = files[key]
        if GZIP:
            with gzip.open(full_path, 'r') as file:
                content_string = file.read().decode('utf-8')
        else:
            with open(full_path, 'r') as file:
                content_string = file.read()
        
        #print(key)

        content = json.loads(content_string)

        if (len(content) != HEIGHT):
            print("ERROR: Unexpected height!", len(content))
            print(full_path)
            exit()
        if (len(content[0]) != WIDTH):
            print("ERROR: Unexpected width!", len(content[0]))
            print(full_path)
            exit()

        if CROP_SETTING == 'crop3':
            w = int(len(content[0]) / 3)
            s1 = crop_w(content, 0, w)
            s2 = crop_w(content, w, w)
            s3 = crop_w(content, 2*w, w)

            dataset.append(s1)
            dataset.append(s2)
            dataset.append(s3)
        else:
            dataset.append(content)
        #print(data)
        #return

        if len(dataset) % 10 == 0:
            print(name, ":\t", len(dataset), "/", total_count, "loaded")

    print("writing", output, "...")
    hf = h5py.File(output, 'w')
    hf.create_dataset(name, data=dataset)
    hf.close()

    print()


def get_keys(keys):
    output1 = []
    output2 = []

    for key in keys:
        if "left" in key:
            alt_key = key.replace("left", "right")
            if alt_key in keys:
                output1.append(key)
                output2.append(alt_key)

    return (output1, output2)


if __name__ == '__main__':
    print("Searching for files...")
    print()

    rgb_t_files = load_files("I:\\mahdi\\sceneflow_rgb_output\\train")
    depth_t_files = load_files('I:\\mahdi\\sceneflow_depth_output\\train')
    rgb_e_files = load_files("I:\\mahdi\\sceneflow_rgb_output\\eval")
    depth_e_files = load_files('I:\\mahdi\\sceneflow_depth_output\\eval')

    rgb_t_keys = [key for key, value in rgb_t_files.items()]
    depth_t_keys = [key for key, value in depth_t_files.items()]
    common_t_keys = list(set(rgb_t_keys).intersection(depth_t_keys))
    random.shuffle(common_t_keys)
    (final_t_keys_02, final_t_keys_03) = get_keys(common_t_keys)

    rgb_e_keys = [key for key, value in rgb_e_files.items()]
    depth_e_keys = [key for key, value in depth_e_files.items()]
    common_e_keys = list(set(rgb_e_keys).intersection(depth_e_keys))
    random.shuffle(common_e_keys)
    (final_e_keys_02, final_e_keys_03) = get_keys(common_e_keys)

    print(len(rgb_t_keys), "keys for rgb train dataset")
    print(len(rgb_e_keys), "keys for rgb eval dataset")
    print(len(depth_t_keys), "keys for depth train dataset")
    print(len(depth_e_keys), "keys for depth eval dataset")
    print(len(final_t_keys_02), "common train left keys")
    print(len(final_t_keys_03), "common train right keys")
    print(len(final_e_keys_02), "common eval left keys")
    print(len(final_e_keys_03), "common eval right keys")
    print()

    e_chunk_count = 1 + int(len(final_e_keys_02) / CHUNK_MAX_SIZE)
    e_keys_02_chunk = chunk(final_e_keys_02, e_chunk_count)
    e_keys_03_chunk = chunk(final_e_keys_03, e_chunk_count)

    print ("Will create", e_chunk_count, "chunks for eval dataset")
    for i in range(e_chunk_count):
        create_h5(e_keys_02_chunk[i], rgb_e_files, 'sceneflow_rgb_eval_02_' + str(i), 'sceneflow_rgb_eval_02_' + str(i) + '.h5')
        create_h5(e_keys_02_chunk[i], depth_e_files, 'sceneflow_depth_eval_02_' + str(i), 'sceneflow_depth_eval_02_' + str(i) + '.h5')
        create_h5(e_keys_03_chunk[i], rgb_e_files, 'sceneflow_rgb_eval_03_' + str(i), 'sceneflow_rgb_eval_03_' + str(i) + '.h5')
        create_h5(e_keys_03_chunk[i], depth_e_files, 'sceneflow_depth_eval_03_' + str(i), 'sceneflow_depth_eval_03_' + str(i) + '.h5')

    t_chunk_count = 1 + int(len(final_t_keys_02) / CHUNK_MAX_SIZE)
    t_keys_02_chunk = chunk(final_t_keys_02, t_chunk_count)
    t_keys_03_chunk = chunk(final_t_keys_03, t_chunk_count)

    print ("Will create", t_chunk_count, "chunks for train dataset")
    for i in range(t_chunk_count):
        create_h5(t_keys_02_chunk[i], rgb_t_files, 'sceneflow_rgb_train_02_' + str(i), 'sceneflow_rgb_train_02_' + str(i) + '.h5')
        create_h5(t_keys_02_chunk[i], depth_t_files, 'sceneflow_depth_train_02_' + str(i), 'sceneflow_depth_train_02_' + str(i) + '.h5')
        create_h5(t_keys_03_chunk[i], rgb_t_files, 'sceneflow_rgb_train_03_' + str(i), 'sceneflow_rgb_train_03_' + str(i) + '.h5')
        create_h5(t_keys_03_chunk[i], depth_t_files, 'sceneflow_depth_train_03_' + str(i), 'sceneflow_depth_train_03_' + str(i) + '.h5')


    print("finished.")
