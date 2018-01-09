import os
import zipfile

from utils import empty_dir


def unzip_with_progress(file_path, path):
    empty_dir(path)
    zf = zipfile.ZipFile(file_path)
    extracted_size = 0

    for contain_file in zf.infolist():
        extracted_size += contain_file.file_size
        zf.extract(contain_file, path)


def zip_path(dir_path):
    path = '/tmp/archive.zip'
    zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()
    return path