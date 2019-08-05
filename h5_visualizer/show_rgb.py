import numpy as np
from PIL import Image
import h5py


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
                (r,g,b) = (values[0], values[1], values[2])
            else:
                (r,g,b) = (values, values, values)
            img.putpixel((i, j), (r, g, b, 255))

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

save_image(data[index], 'show_rgb.png')

