import re
import numpy as np
import sys
from PIL import Image
import colorsys
import os

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

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def save_depth_image(result, image_path):
    height = len(result)
    width = len(result[0])

    maxVal = np.max(result)

    img = Image.new('RGBA', (width, height))

    #print("width:", width)
    #print("height:", height)

    for i in range(width):
        for j in range(height):
            if result[j][i] == -1:
                continue;

            h = 240 * result[j][i] / maxVal;
            (r,g,b) = colorsys.hsv_to_rgb(h / 360,1,1)

            #color = (int)(255 * result[j][i] / maxVal)
            img.putpixel((i, j), (int(r * 255), int(g * 255), int(b * 255), 255))

    img.save(image_path, "PNG")


def normalize(data):
    maxVal = np.max(data)

    print("maxVal:", maxVal)

    data = data / maxVal
    data = 1 - data
    data *= 100

    return data

file_name = input("pfm file?\n")

data, scale, color = readPFM(file_name)

data = normalize(data)

if color:
    print("Not a depth file")
else:
    save_depth_image(data, 'pfm2png_depth.png')
    os.startfile('pfm2png_depth.png')

