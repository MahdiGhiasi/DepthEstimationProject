import h5py;
import os;

def crop_w(image, start, length):
    output = []
    for row in image:
        output.append(row[start:(start+length)])

    return output;

file_name = input("h5 file?\n")

f = h5py.File(file_name, 'r')

a_group_key = list(f.keys())[0]

print()
print('Loading dataset', a_group_key, '...')

data = list(f[a_group_key])


print('Processing...')
data2 = []
for d in data:
    w = int(len(d[0]) / 3)
    s1 = crop_w(d, 0, w)
    s2 = crop_w(d, w, w)
    s3 = crop_w(d, 2*w, w)

    data2.append(s1)
    data2.append(s2)
    data2.append(s3)


output_file_name = "crop3_" + file_name.split(os.path.sep)[-1]
print('Saving...')

hf = h5py.File(output_file_name, 'w')
hf.create_dataset(a_group_key, data=data2)
hf.close()

print("Finished.")
