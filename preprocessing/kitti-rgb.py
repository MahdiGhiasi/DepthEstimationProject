#!/usr/bin/python

SCALE_FACTOR = 12
#TRIM_TOP = 0.32
#TRIM_LEFT = 0.05
TRIMMED_HEIGHT = 18
TRIMMED_WIDTH = 96
THREAD_COUNT = 4

GRAYSCALE = True


from PIL import Image
import numpy as np
import os
import jsonpickle
from multiprocessing import Process

def image_read_and_resize(filename):
    img = Image.open(filename)
    (width, height) = img.size
    img = img.resize((int(width / SCALE_FACTOR), int(height / SCALE_FACTOR)), Image.ANTIALIAS)

    return np.array(img).tolist()

def save_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = np.max(result)

    img = Image.new('RGBA', (width, height))

    #print("width:", width)
    #print("height:", height)

    for i in range(width):
        for j in range(height):
            values = result[j][i];
            if isinstance(values, list):
                if GRAYSCALE:
                    avg = int((values[0] + values[1] + values[2]) / 3)
                    (r,g,b) = (avg, avg, avg)
                else:
                    (r,g,b) = (values[0], values[1], values[2])
            else:
                (r,g,b) = (values, values, values)
            img.putpixel((i, j), (r, g, b, 255))

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
    output_file = open(path, "w")
    output_file.write(json_data)
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

def process(input_path, output_path):
    result = image_read_and_resize(input_path)

    result = trim_top(result)
    result = trim_left(result)

    save_image(result, output_path)

    if GRAYSCALE:
        save_data_grayscale(result, output_path + '.json')
    else:
        save_data(result, output_path + '.json')


def process_files(found_files, input_path, output_path, thread_id):
    counter = 0
    for file in found_files:
        
        file_path = os.path.sep.join(file.split(os.path.sep)[0:-1])
        file_name = file.split(os.path.sep)[-1]

        input_relative_path = file_path[len(input_path):]
        if input_relative_path[0] == os.path.sep:
            input_relative_path = input_relative_path[1:]

        output_file_path = os.path.join(output_path, input_relative_path)
        output_file_fullpath = os.path.join(output_file_path, "i" + file_name)

        #print(output_file_fullpath)

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        process(file, output_file_fullpath)

        counter += 1
        if counter % 10 == 0:
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
        
    #input_path = "H:\\Proje Karshenasi\\Dataset\\KITTI\\2011_09_28_drive_0038_sync\\2011_09_28\\2011_09_28_drive_0038_sync" 
    #input_path = "H:\\Proje Karshenasi\\Dataset\\KITTI\\data_depth_annotated\\" 
    input_path = input("Input files path?")

    output_path = "rgb_output"

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
        process = Process(target=process_files, args=[found_files_chunks[i], input_path, output_path, i])
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("finished")


