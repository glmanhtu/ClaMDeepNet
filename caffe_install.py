from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.utils import *

cuCNN = GoogleFile('0BzL8pCLanAIAUFFvcURwb3EwOHM', 'cudnn-8.0-linux-x64-v6.0.tgz', "/tmp/cudnn-8.0-linux-x64-v6.0.tgz")
google_download = DownloadGoogleDrive()

google_download.download_file_from_google_drive(cuCNN)

execute("chmod +x caffe_install.sh")
execute("./caffe_install.sh")
