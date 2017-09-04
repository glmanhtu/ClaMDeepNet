import sys
import os
import json
from PIL import Image

diff_threshold = 23 #pixel

def get_common_resolution(image_dir):
    common_resolution = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            image = os.path.join(root, f)
            img_size = Image.open(image).size
            flag_has_change = False
            for idx, resolution in enumerate(common_resolution):
                if not has_big_different(resolution['resolution'], img_size):
                    flag_has_change = True
                    common_resolution[idx]['count'] += 1
            common_resolution.append({'resolution' : img_size, 'count' : 1})
    return common_resolution


def filter_big_different_res(common_res, image_dir):
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            image = os.path.join(root, f)
            img_size = Image.open(image).size
            if has_big_different(common_res['resolution'], img_size):
                print "removed %s" % image
                os.remove(image)


def has_big_different(resolution1, resolution2):
    if resolution2[1] == 113:
        resolution2 = resolution2
    diff_width = abs(resolution1[0] - resolution2[0])
    diff_height = abs(resolution1[1] - resolution2[1])
    if diff_width < diff_threshold:
        if diff_height < diff_threshold:
            return False
    return True

if __name__ == '__main__':
    images_dir = sys.argv[1]
    if os.path.isdir(images_dir):
        images_dir = os.path.abspath(images_dir)
        #common_res = get_common_resolution(images_dir)
        most_common_res = {'resolution' : (200, 142), 'count' : 1 }
        filter_big_different_res(most_common_res, images_dir)

    else:
        raise Exception("Please specific image url")
