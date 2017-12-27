import matplotlib

from utils.pycaffe import Caffe

matplotlib.use('Agg')
from network.download_file import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.create_lmdb import CreateLmdb
from utils.make_predictions import *
import os


def heobs_image_classification(template, max_iter, img_width, img_height, gpu_id, lr, stepsize, batchsize_train, batchsize_test):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    classes = ["being", "heritage", "scenery"]
    train_zip = GoogleFile('0BzL8pCLanAIAd0hBV2NUVHpmckE', workspace('data/heobs_large_dataset.zip'))

    caffe = Caffe()
    google_download = DownloadGoogleDrive()

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
                     test_path, img_width, img_height)

    mean_proto = workspace("data/mean.binaryproto")

    caffe.compute_image_mean("lmdb", train_lmdb_path, mean_proto)
    caffe.compute_image_mean("lmdb", validation_lmdb_path, mean_proto)

    solver_mode = constant.CAFFE_SOLVER
    if "CAFFE_SOLVER" in os.environ:
        solver_mode = os.environ['CAFFE_SOLVER']

    py_render_template("template/" + template + "/caffenet_train.template", caffe_train_model, mean_file=mean_proto,
                       train_lmdb=train_lmdb_path, validation_lmdb=validation_lmdb_path,
                       batchsize_train=batchsize_train,
                       batchsize_test=batchsize_test,
                       num_output=len(classes))
    py_render_template("template/" + template + "/caffenet_solver.template", caffe_solver,
                       caffe_train_model=caffe_train_model,
                       max_iter=max_iter,
                       snapshot_prefix=snapshot_prefix,
                       learning_rate=lr,
                       stepsize=stepsize,
                       solver_mode=solver_mode)

    print "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"

    print "\nWeb app started"

    print "\nStarting to train"
    caffe.train(caffe_solver, caffe_log, gpu_id)

    print "\nTrain completed"

    print "\nStarting to test"

    print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    py_render_template("template/" + template + "/caffenet_deploy.template", caffe_deploy,
                       num_output=len(classes), img_width=img_width, img_height=img_height)

    mean_data = read_mean_data(mean_proto)
    net = read_model_and_weight(caffe_deploy, snapshot_prefix + "_iter_" + str(max_iter) + ".caffemodel")
    transformer = image_transformers(net, mean_data)
    prediction = making_predictions(workspace("data/extracted/test"), transformer, net, img_width, img_height)

    empty_dir("result")

    export_to_csv(prediction, workspace("result/test_result.csv"))
    export_data(prediction, workspace("data/extracted/test"), workspace("result/data"))

    draw_curve(caffe_log, workspace("result/curve.png"), template)
    shutil.copyfile(caffe_log, workspace("result/caffe_train.log"))

    print "\n\n-------------------------FINISH------------------------------------\n\n"

    print "\nTest completed"


if __name__ == '__main__':

    heobs_image_classification("googlenet", 4000, 224, 224, "0", 0.01, 3000, 32, 50);
