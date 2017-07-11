import urllib2
import os
from utils.utils import save_checksum
from utils.percent_visualize import print_progress


def download_file(url, destination_path):

    destination_dir = os.path.dirname(destination_path)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(destination_path, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        f_buffer = u.read(block_sz)
        if not f_buffer:
            break

        file_size_dl += len(f_buffer)
        f.write(f_buffer)
        print_progress(file_size_dl, file_size, "Progress:", "Complete", 2, 50)

    f.close()
    save_checksum(destination_path)
