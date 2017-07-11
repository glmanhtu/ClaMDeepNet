"""Because we have alot of images, we're not be able to classify all of them (>100.000 images)
So, we'll select random 1000 images, classify them & use them to train our neural network
After that, we using trained network to classify the other images"""

from get_all_photos import *
from random import randint
from shutil import copyfile
import os

TOTAL_SAMPLE = 1000
LABELED_IMAGES = os.path.join("images", "sample", "unknown")


if __name__ == '__main__':
    path, dirs, files = os.walk(SAVE_IMAGE_PATH).next()
    num_files = len(files)
    random_imgs = []


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

    for root, dirs, files in os.walk(SAVE_IMAGE_PATH):
        for f in files:
            image = os.path.join(root, f)
            if current_index in random_imgs:
                copyfile(image, os.path.join(LABELED_IMAGES, f))
            print_progress(current_index, num_files, "Progress:", "Complete", 2, 50)
            current_index += 1

