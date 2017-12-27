import urllib2
import os

from download_google_drive import DownloadGoogleDrive
from google_file import GoogleFile
from utils.utils import save_checksum
from utils.percent_visualize import print_progress
import re

google_download = DownloadGoogleDrive()


def download_file_strategy(url, destination_path):
    if "drive.google.com" in url:
        file_id = re.findall('id=([^\/]*)', url)[0]
        train_zip = GoogleFile(file_id, destination_path)
        google_download.download_file_from_google_drive(train_zip)
    else:
        download_file(url, destination_path)


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
