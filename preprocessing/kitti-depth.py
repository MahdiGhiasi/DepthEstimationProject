#!/usr/bin/python

SCALE_FACTOR = 2
#TRIM_TOP = 0.32
#TRIM_LEFT = 0.05
TRIMMED_HEIGHT = 30 * 4
TRIMMED_WIDTH = 144 * 4
THREAD_COUNT = 4

from PIL import Image
import numpy as np
import os
import colorsys
from multiprocessing import Process
import jsonpickle

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    #print(np.max(depth_png))

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth.tolist()

def save_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = 85

    img = Image.new('RGBA', (width, height))

    #print("width:", width)
    #print("height:", height)

    for i in range(width):
        for j in range(height):
            if result[j][i] == -1:
                continue;

            h = 240 * min(result[j][i] / maxVal, 1);
            (r,g,b) = colorsys.hsv_to_rgb(h / 360,1,1)

            #color = (int)(255 * result[j][i] / maxVal)
            img.putpixel((i, j), (int(r * 255), int(g * 255), int(b * 255), 255))

    img.save(image_path, "PNG")

def scale(result, factor):
    height = len(result)
    width = len(result[0])

    result2 = [];

    for i in range(height):
        if i % factor != 0:
            continue
        if i + factor >= height:
            continue
        row = []
        for j in range(width):
            if j % factor != 0:
                continue
            
            if j + factor >= width:
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

def fill_sky(result):
    height = len(result)
    width = len(result[0])
    maxVal = np.max(result).item()

    for i in range(width):
        for j in range(height):
            if result[j][i] != -1:
                break;

            result[j][i] = maxVal
            
    return result


def fill_empty_rows(result):
    height = len(result)
    width = len(result[0])
    maxVal = np.max(result).item()

    for i in range(height):
        isEmpty = True
        for j in range(width):
            if result[i][j] != -1:
                isEmpty = False
        if isEmpty:
            for j in range(width):
                result[i][j] = maxVal
            
    return result



def fill_holes(result):
    height = len(result)
    width = len(result[0])

    for i in range(height):
        for j in range(width):
            if result[i][j] != -1:
                continue;

            search_size = 1

            count = 0
            avg = 0
            while count == 0:
                for x in range(-search_size, search_size + 1):
                    for y in range(-search_size, search_size + 1):
                        ii = i + x
                        jj = j + y
                        if ii < 0 or ii >= height or jj < 0 or jj >= width:
                            continue

                        if result[ii][jj] != -1:
                            avg += result[ii][jj]
                            count += 1
                
                search_size += 1
            
            avg /= count
            result[i][j] = avg

    return result

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

def preprocess(input_path, output_path):
    result = depth_read(input_path)


    result = scale(result, SCALE_FACTOR)
    #result = fill_sky(result)
    result = fill_empty_rows(result)
    result = fill_holes(result)

    result = trim_top(result)
    result = trim_left(result)

    save_image(result, output_path)
    save_data(result, output_path + '.json')


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

    output_path = "depth_output"

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
