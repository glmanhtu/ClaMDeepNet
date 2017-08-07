import sys
import os
from PIL import Image


def get_lowest_resolution(image_dir):
    min_width, min_height = 9999, 9999
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            image = os.path.join(root, f)
            width, height = Image.open(image).size
            min_width = lower(min_width, width)
            min_height = lower(min_height, height)
    return min_width, min_height


def lower(first, last):
    if first < last:
        return first
    return last

if __name__ == '__main__':
    images_dir = sys.argv[1]
    if os.path.isdir(images_dir):
        images_dir = os.path.abspath(images_dir)
        w, h = get_lowest_resolution(images_dir)
        print "min w %d, min h %d" %  (w, h)
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                image = os.path.join(root, f)
                print image
                im = Image.open(image).resize((w, h), Image.BICUBIC)
                im.save(image)

    else:
        raise Exception("Please specific image url")
