"""Because we have alot of inaccessible images so, this file will filter all of them out of our dataset"""

from get_all_photos import *
import imghdr
import os
import sys

if __name__ == '__main__':
    img_dir = sys.argv[1]
    path, dirs, files = os.walk(img_dir).next()
    num_files = len(files)
    current_progress = 0
    deleted_count = 0
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            image = os.path.join(root, f)
            if imghdr.what(image) != 'jpeg':
                os.remove(image)
                deleted_count += 1
            print_progress(current_progress, num_files, "Progress:", "Complete", 2, 50)
            current_progress += 1
    print "Total images: %d" % num_files
    print "Deleted broken images : %d -> %f percent" % (deleted_count, round((deleted_count / float(num_files)) * 100, 2))
    image_left = num_files - deleted_count
    print "Images left: %d -> %f percent" % (image_left, round((image_left / float(num_files)) * 100, 2))
