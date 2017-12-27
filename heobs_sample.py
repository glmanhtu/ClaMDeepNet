import matplotlib

matplotlib.use('Agg')
from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.create_lmdb import CreateLmdb
from utils.pycaffe import Caffe
import shutil
from curve import *
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

classes = ["being", "heritage", "scenery"]

google_download = DownloadGoogleDrive()
train_zip = GoogleFile('0BzL8pCLanAIAd0hBV2NUVHpmckE', workspace('data/heobs_large_dataset.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

print "Starting download train file"
google_download.download_file_from_google_drive(train_zip)
print "Finish"

print "Extracting train zip file"
unzip_with_progress(train_zip.file_path, workspace("data/extracted"))
print "Finish"

train_lmdb_path = workspace("data/extracted/train_lmdb")

validation_lmdb_path = workspace("data/extracted/validation_lmdb")

test_path = workspace("data/extracted/test")

# Will create after render
caffe_train_model = workspace("caffe_model/caffenet_train.prototxt")

# Will create after render
caffe_solver = workspace("caffe_model/caffenet_solver.prototxt")

caffe_log = workspace("caffe_model/caffe_train.log")

caffe_deploy = workspace("caffe_model/caffenet_deploy.prototxt")

snapshot_prefix = workspace("caffe_model/snapshot")

lmdb = CreateLmdb()
lmdb.create_lmdb(workspace("data/extracted/heobs_large_dataset"), train_lmdb_path, validation_lmdb_path, classes,
                 test_path)

mean_proto = workspace("data/mean.binaryproto")

caffe = Caffe()
caffe.compute_image_mean("lmdb", train_lmdb_path, mean_proto)
caffe.compute_image_mean("lmdb", validation_lmdb_path, mean_proto)

solver_mode = constant.CAFFE_SOLVER
if "CAFFE_SOLVER" in os.environ:
    solver_mode = os.environ['CAFFE_SOLVER']

py_render_template("template/" + template() + "/caffenet_train.template", caffe_train_model, mean_file=mean_proto,
                   train_lmdb=train_lmdb_path, validation_lmdb=validation_lmdb_path,
                   num_output=Constant.NUMBER_OUTPUT)
py_render_template("template/" + template() + "/caffenet_solver.template", caffe_solver,
                   caffe_train_model=caffe_train_model,
                   snapshot_prefix=snapshot_prefix,
                   solver_mode=solver_mode)

print "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"

print "\nWeb app started"

print "\nStarting to train"
caffe.train(caffe_solver, caffe_log)

print "\nTrain completed"

print "\nStarting to test"

print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"

os.chdir(os.path.dirname(os.path.realpath(__file__)))

py_render_template("template/" + template() + "/caffenet_deploy.template", caffe_deploy,
                   num_output=Constant.NUMBER_OUTPUT, img_width=Constant.IMAGE_WIDTH, img_height=Constant.IMAGE_HEIGHT)

mean_data = read_mean_data(mean_proto)
net = read_model_and_weight(caffe_deploy, snapshot_prefix + "_iter_4000.caffemodel")
transformer = image_transformers(net, mean_data)
prediction = making_predictions(workspace("data/extracted/test"), transformer, net)

empty_dir("result")

export_to_csv(prediction, workspace("result/test_result.csv"))
export_data(prediction, workspace("data/extracted/test"), workspace("result/data"))

draw_curve(caffe_log, workspace("result/curve.png"))
shutil.copyfile(caffe_log, workspace("result/caffe_train.log"))

print "\n\n-------------------------FINISH------------------------------------\n\n"

print "\nTest completed"
