#!/usr/bin/python

SCALE_FACTOR = 4.5
# --> Total scale factor = 9/2 = 4.5
#TRIM_TOP = 0.32
#TRIM_LEFT = 0.05
TRIMMED_HEIGHT = 120
TRIMMED_WIDTH = 192
THREAD_COUNT = 4

GRAYSCALE = False


from PIL import Image
import numpy as np
import os
import jsonpickle
from multiprocessing import Process
import gzip
import re
import colorsys

def save_rgb_image(result, image_path):
    height = len(result)
    width = len(result[0])

    output = []
    for i in range(height):
        row = []
        for j in range(width):
            values = result[i][j];
            if isinstance(values, list):
                if GRAYSCALE:
                    avg = int((values[0] + values[1] + values[2]) / 3)
                    (r,g,b) = (avg, avg, avg)
                else:
                    (r,g,b) = (values[0], values[1], values[2])
            else:
                (r,g,b) = (values, values, values)

            row.append([int(r * 255), int(g * 255), int(b * 255), 255])
        output.append(row)

    img = Image.fromarray(np.asarray(output).astype(np.uint8), mode='RGBA')
    
    img.save(image_path, "PNG")

def trim_top(result):
    height = len(result)
    new_height = TRIMMED_HEIGHT
    while len(result) > new_height:
        result.pop(0)
    return result

def trim_left(result):
    width = len(result[0])
    new_width = TRIMMED_WIDTH
    while len(result[0]) > new_width:
        for i in range(len(result)):
            result[i].pop(0)
    return result

def save_data(data, path):
    json_data = jsonpickle.encode(data)
    #print(json_data)
    output_file = gzip.open(path, "w")
    output_file.write(json_data.encode())
    output_file.close()

def save_data_grayscale(data, path):
    data2 = []

    for row in data:
        x = []
        for d in row:
            if isinstance(d, list):
                x.append(int((d[0] + d[1] + d[2]) / 3))
            else:
                save_data(data, path)
                return
        data2.append(x)

    save_data(data2, path)


def image_read_and_resize(filename):
    img = Image.open(filename)
    (width, height) = img.size
    img = img.resize((int(width / SCALE_FACTOR), int(height / SCALE_FACTOR)), Image.ANTIALIAS)

    return np.array(img).tolist()

def preprocess(input_path, output_path):
    data = image_read_and_resize(input_path)

    data = trim_top(data)
    data = trim_left(data)

    save_rgb_image(data, output_path)
    if GRAYSCALE:
        save_data_grayscale(data, output_path + '.json.gz')
    else:
        save_data(data, output_path + '.json.gz')


def preprocess_files(found_files, input_path, output_path, thread_id):
    counter = 0
    for file in found_files:        
        file_path = os.path.sep.join(file.split(os.path.sep)[0:-1])
        file_name = file.split(os.path.sep)[-1]

        input_relative_path = file_path[len(input_path):]
        if input_relative_path[0] == os.path.sep:
            input_relative_path = input_relative_path[1:]

        output_file_path = os.path.join(output_path, input_relative_path)
        output_file_fullpath = os.path.join(output_file_path, "i" + file_name)

        try:
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)
        except:
            pass

        preprocess(file, output_file_fullpath)

        counter += 1
        if counter % 1 == 0:
            print(counter, "files processed in thread", thread_id)


def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



if __name__ == '__main__':
    
    #input_path = "H:\\Proje Karshenasi\\Dataset\\KITTI\\data_depth_annotated\\train\\2011_09_28_drive_0038_sync\\proj_depth\\groundtruth" 
    #input_path = "H:\\Proje Karshenasi\\Dataset\\KITTI\\data_depth_annotated\\" 
    input_path = input("Input files path?\n")

    output_path = "sceneflow_rgb_output"

    print("Searching for files...")
    found_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".png"):
                fullpath = os.path.join(root, file)
                found_files.append(fullpath)
                if len(found_files) % 1000 == 0:
                    print(len(found_files), "files found")
        

    print(len(found_files), "files found")
    print()
    print("Processing in", THREAD_COUNT, "parallel threads...")

    found_files_chunks = chunk(found_files, THREAD_COUNT)

    processes = []
    for i in range(THREAD_COUNT):
        process = Process(target=preprocess_files, args=[found_files_chunks[i], input_path, output_path, i])
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("finished")
