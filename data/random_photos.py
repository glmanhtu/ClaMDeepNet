from get_all_photos import *
from random import randint
from shutil import copyfile
import os
import sys

TOTAL_SAMPLE = 1000
RANDOM_IMAGES = os.path.join("images", "sample", "unknown")


if __name__ == '__main__':
    img_dir = sys.argv[1]
    path, dirs, files = os.walk(img_dir).next()
    num_files = len(files)
    random_imgs = []

    if not os.path.isdir(RANDOM_IMAGES):
        os.makedirs(RANDOM_IMAGES)

    def add_random_numb():
        random_numb = randint(0, num_files)
        if random_numb in random_imgs:
            add_random_numb()
        else:
            random_imgs.append(random_numb)


    for i in range(TOTAL_SAMPLE):
        add_random_numb()

    current_index = 0
    copied = 0

    for root, dirs, files in os.walk(img_dir):
        for f in files:
            image = os.path.join(root, f)
            if current_index in random_imgs:
                copyfile(image, os.path.join(RANDOM_IMAGES, f))
            print_progress(current_index, num_files, "Progress:", "Complete", 2, 50)
            current_index += 1

