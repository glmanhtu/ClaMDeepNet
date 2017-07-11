"""All Images stored in cloud, this file will fetch all of them & save to SAVE_IMAGE_PATH
According to photo list described in FILE_NAME"""

import re
import urllib
import os

from utils.print_progress import print_progress

FILE_NAME = "photos.txt"
BASE_URL = "https://cdn.heobs.org/photo/"
SAVE_IMAGE_PATH = os.path.join("images", "dataset")

if __name__ == '__main__':
    print "Fetching images"

    print "Reading image name file, %s" % FILE_NAME

    num_lines = sum(1 for line in open(FILE_NAME))
    images = []
    file_name_count = 0
    line_count = 0
    with open(FILE_NAME) as f:
        for line in f:
            m = re.search("\s*([\w-]+)\s+\|\s+([^|]+)", line)
            line_count = line_count + 1
            if m:
                url_path = m.group(1)
                if url_path in images:
                    file_name_count = file_name_count + 1
                    continue
                else:
                    images.append(url_path)
                image_name = str(url_path)
                url = BASE_URL + url_path
                file_path = SAVE_IMAGE_PATH + "/" + image_name + ".jpg"
                urllib.urlretrieve(url, file_path)
                print_progress(line_count, num_lines, "Progress:", "Complete", 2, 50)
                file_name_count = file_name_count + 1

