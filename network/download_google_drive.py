import re
from sys import getsizeof
from dependencies.requests.requests import sessions
from utils.percent_visualize import print_progress
from utils.utils import *


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def get_file_size(file_name, security_content):
    pattern = r"<\/a>\s*\(([^)]*)\)"
    match_obj = re.search(pattern, security_content, re.M | re.UNICODE)
    if match_obj is None:
        return None
    file_size = match_obj.group(1)
    return human_2_bytes(file_size)


class DownloadGoogleDrive(object):
    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768

    def download_file_from_google_drive(self, google_file):

        if file_already_exists(google_file.file_path):
            print ("File %s already downloaded & verified" % google_file.file_path)
            return

        g_session = sessions.Session()

        response = g_session.get(self.URL, params={'id': google_file.file_id}, stream=True)
        file_size = get_file_size(google_file.file_name, response.content)
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



