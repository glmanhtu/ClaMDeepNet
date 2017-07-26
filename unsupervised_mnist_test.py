from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.make_predictions import *

google_download = DownloadGoogleDrive()

set_workspace("data/unsupervised_mnist")

test_zip = GoogleFile('0BzL8pCLanAIARUFPNTNqa1lQekk',
                       'mnist_train_data.zip', workspace('data/mnist_train_data.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

print "Starting download test file"
google_download.download_file_from_google_drive(test_zip)
print "Finish"

print "Extracting test zip file"
unzip_with_progress(test_zip.file_path, workspace("data"))
print "Finish"

print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"

mean_proto = workspace("data/mean.binaryproto")

caffe_deploy = workspace("caffe_model/caffenet_deploy.prototxt")

py_render_template("template/caffenet_deploy.template", caffe_deploy,
                   num_output=Constant.NUMBER_OUTPUT)

net = read_model_and_weight(caffe_deploy, "/workspace/init.caffemodel")
transformer = image_transformers(net, None)
prediction = making_predictions(workspace("data/trainingSet"), transformer, net)

export_to_csv(prediction, workspace("result/test_result.csv"))

print "\n\n-------------------------FINISH------------------------------------\n\n"
