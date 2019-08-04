import os
import jsonpickle
from threading import Thread
import h5py


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

# def chunk(seq, num):
#     avg = len(seq) / float(num)
#     out = []
#     last = 0.0

#     while last < len(seq):
#         out.append(seq[int(last):int(last + avg)])
#         last += avg

#     return out

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
            if file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                found_files.append(full_path)

    for full_path in found_files:
        path_chunks = full_path.split(os.path.sep)

        key = ""
        for s in path_chunks:
            if s.endswith("_sync"):
                key += s + "__"
            elif s.startswith("image_"):
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

    print("creating dataset", name, "...")
    for key in keys:
        full_path = files[key]
        with open(full_path, 'r') as file:
            content_string = file.read()
        
        #print(key)

        content = jsonpickle.decode(content_string)

        dataset.append(content)
        #print(data)
        #return

        if len(dataset) % 100 == 0:
            print("\r", len(dataset), "/", len(keys), "loaded", end='')

    print()
    print("writing", output, "...")
    hf = h5py.File(output, 'w')
    hf.create_dataset(name, data=dataset)
    hf.close()

    print()



print("Searching for files...")
print()

rgb_t_files = load_files("C:\\Users\\Mahdi\\Projects\\DepthEstimationProject\\preprocessing\\rgb_output\\train")
depth_t_files = load_files('H:\\Proje Karshenasi\\Dataset\\KITTI\\depth_output\\train')
rgb_e_files = load_files("C:\\Users\\Mahdi\\Projects\\DepthEstimationProject\\preprocessing\\rgb_output\\val")
depth_e_files = load_files('H:\\Proje Karshenasi\\Dataset\\KITTI\\depth_output\\val')

rgb_t_keys = [key for key, value in rgb_t_files.items()]
depth_t_keys = [key for key, value in depth_t_files.items()]
common_t_keys = list(set(rgb_t_keys).intersection(depth_t_keys))

rgb_e_keys = [key for key, value in rgb_e_files.items()]
depth_e_keys = [key for key, value in depth_e_files.items()]
common_e_keys = list(set(rgb_e_keys).intersection(depth_e_keys))

print(len(rgb_t_keys), "keys for rgb train dataset")
print(len(rgb_e_keys), "keys for rgb eval dataset")
print(len(depth_t_keys), "keys for depth train dataset")
print(len(depth_e_keys), "keys for depth eval dataset")
print(len(common_t_keys), "common train keys")
print(len(common_e_keys), "common eval keys")
print()

create_h5(common_t_keys, rgb_t_files, 'rgb_train', 'rgb_train.h5')
create_h5(common_t_keys, depth_t_files, 'depth_train', 'depth_train.h5')
create_h5(common_e_keys, rgb_e_files, 'rgb_eval', 'rgb_eval.h5')
create_h5(common_e_keys, depth_e_files, 'depth_eval', 'depth_eval.h5')
