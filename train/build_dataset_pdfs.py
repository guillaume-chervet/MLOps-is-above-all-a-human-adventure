from PIL import Image

import glob
import shutil
from random import random
import uuid
import random

root_dir = './dogs-vs-cats/train'
# root_dir needs a trailing slash (i.e. /root/dir/)
files = glob.iglob(root_dir + '**/*', recursive=True)

count = 0
for filename in files:
     print(filename)
     count = count+1
     shutil.copy(filename, './imgs_guid/' + str(uuid.uuid4()) +'.jpg')

print(str(count))

root_dir = './imgs_guid'
# root_dir needs a trailing slash (i.e. /root/dir/)
files = glob.iglob(root_dir + '**/*', recursive=True)
files_array = []
for filename in files:
    # seed random number generator
    files_array.append(filename)

count = len(files_array)
def get_image(files_array):
    print(len(files_array))
    max = len(files_array)
    selected_index = int(random.randrange(0, max))
    print(selected_index)
    filepath = files_array[selected_index]
    print(filepath)
    files_array.pop(selected_index)
    return files_array, filepath


while len(files_array) > 0:
    images =[]
    number_pages = int(random.randrange(0, 10))
    for i in range(number_pages):
        files_array, filepath = get_image(files_array)
        img = Image.open(filepath).convert("RGB")
        images.append(img)
    img.save('./pdfs/' + str(uuid.uuid4()) + ".pdf", save_all=True, append_images=images)
    #im1 = Image.open(filename).convert("RGB")
    #im2 = Image.open("2.jpg").convert("RGB")
    #im3 = Image.open("3.jpg").convert("RGB")
    #images = [im1]
    #im1.save( str(uuid.uuid4()) + ".pdf", save_all=True, append_images=images)