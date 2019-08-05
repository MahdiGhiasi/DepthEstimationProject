import numpy as np
from PIL import Image
import h5py
import colorsys


def save_image(result, image_path):
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


#file_name = "C:\\Users\\Mahdi\\Projects\\DepthEstimationProject\\create_h5_dataset\\rgb_train_0.h5"
file_name = input("h5 file?\n")

f = h5py.File(file_name, 'r')

#print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

print()
print('Loading dataset', a_group_key, '...')

data = list(f[a_group_key])

print('Dataset contains', len(data), 'entries. Which one do you want?')
index = int(input())

save_image(data[index], 'show_depth.png')

