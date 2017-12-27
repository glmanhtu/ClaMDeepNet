import urllib2
import os
from sys import getsizeof

from dependencies.requests.requests import sessions
from google_file import GoogleFile
from utils.utils import save_checksum, file_already_exists, human_2_bytes
from utils.percent_visualize import print_progress
import re


class DownloadGoogleDrive(object):
    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768

    def download_file_from_google_drive(self, google_file):

        if file_already_exists(google_file.file_path):
            print ("File %s already downloaded & verified" % google_file.file_path)
            return

        g_session = sessions.Session()

        response = g_session.get(self.URL, params={'id': google_file.file_id}, stream=True)
        file_size = get_file_size(response.content)
        if file_size is None:
            file_size = getsizeof(response.content)
            self.save_response_content(response, google_file.file_path, file_size)
        else:
            token = get_confirm_token(response)

            if token:
                params = {'id': google_file.file_id, 'confirm': token}
                response = g_session.get(self.URL, params=params, stream=True)

            self.save_response_content(response, google_file.file_path, file_size)

    def save_response_content(self, response, destination, file_size):
        print "Downloading %s" % destination
        dl = 0
        total_length = file_size
        with open(destination, "wb") as f:
            for chunk in response.iter_content(self.CHUNK_SIZE):
                dl += len(chunk)
                if chunk:
                    f.write(chunk)
                    print_progress(dl, total_length, "Progress:", "Complete", 2, 50)
        save_checksum(destination)


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


def get_file_size(security_content):
    pattern = r"<\/a>\s*\(([^)]*)\)"
    match_obj = re.search(pattern, security_content, re.M | re.UNICODE)
    if match_obj is None:
        return None
    file_size = match_obj.group(1)
    return human_2_bytes(file_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None