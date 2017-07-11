import zipfile
from percent_visualize import print_progress


def unzip_with_progress(file_path, path):
    zf = zipfile.ZipFile(file_path)

    uncompress_size = sum((file.file_size for file in zf.infolist()))

    extracted_size = 0

    for contain_file in zf.infolist():
        extracted_size += contain_file.file_size
        print_progress(extracted_size, uncompress_size, "Progress:", "Complete", 2, 50)
        zf.extract(contain_file, path)