#!/usr/bin/python

SCALE_UP_FACTOR = 2
SCALE_DOWN_FACTOR = 9
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


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        print(header)
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale, color


    
def normalize(data):
    data = 35 * 32 / data
    data = 3 * data / 8
    data = np.minimum(data, 85)

    #print(data)    

    return data

def save_depth_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = np.max(result)

    #print("width:", width)
    #print("height:", height)

    output = []
    for i in range(height):
        row = []
        for j in range(width):
            h = 240 * result[i][j] / maxVal;
            (r,g,b) = colorsys.hsv_to_rgb(h / 360,1,1)

            #color = (int)(255 * result[j][i] / maxVal)
            row.append([int(r * 255), int(g * 255), int(b * 255), 255])
        output.append(row)

    img = Image.fromarray(np.asarray(output).astype(np.uint8), mode='RGBA')
    
    img.save(image_path, "PNG")

def scale_up_image(result, factor):
    result2 = []

    for row in result:
        new_row = []
        for item in row:
            for i in range(factor):
                new_row.append(item)
        
        for i in range(factor):
            result2.append(new_row)
    
    return result2

def scale_down_image(result, factor):
    height = len(result)
    width = len(result[0])

    result2 = [];

    for i in range(height):
        if i % factor != 0:
            continue
        if i + factor > height:
            continue
        row = []
        for j in range(width):
            if j % factor != 0:
                continue
            
            if j + factor > width:
                continue

            count = 0
            avg = 0
            for x in range(factor):
                for y in range(factor):
                    if result[i + x][j + y] == -1:
                        continue
                    avg += result[i + x][j + y]
                    count += 1
            
            if count == 0:
                row.append(-1)
                continue
            else:
                avg /= count
                row.append(avg)

        result2.append(row)

    return result2

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

def preprocess(input_path, output_path):
    data, scale, color = readPFM(input_path)

    if color:
        print("Not a depth pfm file.")
        return    

    data = normalize(data)

    maxVal = np.max(data)
    #print(input_path, ":", maxVal)

    data = scale_up_image(data, SCALE_UP_FACTOR)
    data = scale_down_image(data, SCALE_DOWN_FACTOR)
    data = trim_top(data)
    data = trim_left(data)

    save_depth_image(data, output_path)
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
        output_file_fullpath = os.path.join(output_file_path, "p" + file_name)

        output_file_fullpath = output_file_fullpath.replace('.pfm', '.png')
        #print(output_file_fullpath)

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

    output_path = "sceneflow_depth_output"

    print("Searching for files...")
    found_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".pfm"):
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
