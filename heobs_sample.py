import matplotlib
matplotlib.use('Agg')
from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.create_lmdb import CreateLmdb
from utils.pycaffe import Caffe
from utils.make_predictions import *
from curve import *
import heobs_sample_test

google_download = DownloadGoogleDrive()

set_workspace(os.path.join("workspace", "heobs_sample"))

shutil.rmtree(workspace(""))

train_zip = GoogleFile('0BzL8pCLanAIAd0hBV2NUVHpmckE',
                       'heobs_large_dataset.zip', workspace('data/heobs_large_dataset.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

print "Starting download train file"
google_download.download_file_from_google_drive(train_zip)
print "Finish"

print "Extracting train zip file"
unzip_with_progress(train_zip.file_path, workspace("data"))
print "Finish"


train_lmdb_path = workspace("data/train_lmdb")
validation_lmdb_path = workspace("data/validation_lmdb")
test_path = workspace("data/test")

lmdb = CreateLmdb()
classes = ["being", "heritage", "scenery"]
lmdb.create_lmdb(workspace("data/heobs_large_dataset"), train_lmdb_path, validation_lmdb_path, classes, test_path)

mean_proto = workspace("data/mean.binaryproto")

caffe = Caffe()
caffe.compute_image_mean("lmdb", train_lmdb_path, mean_proto)
caffe.compute_image_mean("lmdb", validation_lmdb_path, mean_proto)

# Will create after render
caffe_train_model = workspace("caffe_model/caffenet_train.prototxt")

# Will create after render
caffe_solver = workspace("caffe_model/caffenet_solver.prototxt")

solver_mode = constant.CAFFE_SOLVER
if "CAFFE_SOLVER" in os.environ:
    solver_mode = os.environ['CAFFE_SOLVER']

py_render_template("template/" + template() + "/caffenet_train.template", caffe_train_model, mean_file=mean_proto,
                   train_lmdb=train_lmdb_path, validation_lmdb=validation_lmdb_path,
                   num_output=Constant.NUMBER_OUTPUT)
py_render_template("template/" + template() + "/caffenet_solver.template", caffe_solver, caffe_train_model=caffe_train_model,
                   snapshot_prefix=workspace("caffe_model/snapshot"),
                   solver_mode=solver_mode)

caffe_log = workspace("caffe_model/caffe_train.log")

print "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"

run_thread_app()

caffe.train(caffe_solver, caffe_log)

heobs_sample_test.test()