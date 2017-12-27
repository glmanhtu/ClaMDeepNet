import hashlib
import os
import re
import urllib2
from sys import getsizeof

from dependencies.requests.requests import sessions
from google_file import GoogleFile


class DownloadGoogleDrive(object):
    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768

    def download_file_from_google_drive(self, google_file):

        if file_already_exists(google_file.file_path):
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
        dl = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(self.CHUNK_SIZE):
                dl += len(chunk)
                if chunk:
                    f.write(chunk)
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

    u = urllib2.urlopen(url)
    f = open(destination_path, 'wb')

    file_size_dl = 0
    block_sz = 8192
    while True:
        f_buffer = u.read(block_sz)
        if not f_buffer:
            break

        file_size_dl += len(f_buffer)
        f.write(f_buffer)

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


def save_checksum(file_path):
    read_size = 1024  # You can make this bigger
    checksum1 = hashlib.md5()
    with open(file_path, 'rb') as f:
        data = f.read(read_size)
        while data:
            checksum1.update(data)
            data = f.read(read_size)
    checksum1 = checksum1.hexdigest()
    with open(file_path + ".checksum", "w") as text_file:
        text_file.write(checksum1)


def file_already_exists(file_path):
    if os.path.isfile(file_path):
        checksum_file = file_path + ".checksum"
        if os.path.isfile(checksum_file):
            checksum_original = open(checksum_file, 'r').read()
            checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if checksum_original == checksum:
                return True
            os.remove(checksum_file)
        os.remove(file_path)
    return False


def human_2_bytes(s):
    """
    >>> human2bytes('1M')
    1048576
    >>> human2bytes('1G')
    1073741824
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])