import Queue
import traceback

from network.download_file import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.create_lmdb import CreateLmdb
from utils.make_predictions import *
from utils.pycaffe import PyCaffe
from utils.utils import *
from utils.workspace import Workspace
from utils.zip_utils import unzip_with_progress


def heobs_image_classification(template, max_iter, img_width, img_height, gpu_id, lr, stepsize, batchsize_train,
                               batchsize_test, trained_model, ws, queue, test_id):
    # type: (str, int, int, int, str, float, int, int, int, str, Workspace, Queue.Queue, int) -> None

    try:
        queue.put(("update", test_id, 1, 100, "starting..."))

        queue.put(("log", test_id, "Working dir: %s" % ws.workspace("")))
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        classes = ["being", "heritage", "scenery"]

        train_lmdb_path = ws.workspace("data/extracted/train_lmdb")

        validation_lmdb_path = ws.workspace("data/extracted/validation_lmdb")

        test_path = ws.workspace("data/extracted/test")

        # Will create after render
        caffe_train_model = ws.workspace("caffe_model/caffenet_train.prototxt")

        # Will create after render
        caffe_solver = ws.workspace("caffe_model/caffenet_solver.prototxt")

        caffe_log = ws.workspace("caffe_model/caffe_train.log")

        caffe_deploy = ws.workspace("caffe_model/caffenet_deploy.prototxt")

        snapshot_prefix = ws.workspace("caffe_model/snapshot")

        mean_proto = ws.workspace("data/mean.binaryproto")

        pycaffe = PyCaffe()

        if not os.path.isfile(caffe_log):

            train_zip = GoogleFile('0BzL8pCLanAIAd0hBV2NUVHpmckE', ws.workspace('data/heobs_large_dataset.zip'))

            queue.put(("update", test_id, 2, 100, "downloading dataset..."))
            google_download = DownloadGoogleDrive()

            queue.put(("log", test_id, "\n\n------------------------PREPARE PHRASE----------------------------\n\n"))

            queue.put(("log", test_id, "Starting download train file"))
            google_download.download_file_from_google_drive(train_zip)
            queue.put(("log", test_id, "Finish"))

            queue.put(("log", test_id, "Extracting train zip file"))
            queue.put(("update", test_id, 10, 100, "extracting dataset..."))
            unzip_with_progress(train_zip.file_path, ws.workspace("data/extracted"))
            queue.put(("log", test_id, "Finish"))

            queue.put(("update", test_id, 15, 100, "creating lmdb..."))
            lmdb = CreateLmdb()
            lmdb.create_lmdb(ws.workspace("data/extracted/heobs_large_dataset"), train_lmdb_path, validation_lmdb_path,
                             classes,
                             test_path, img_width, img_height)

            queue.put(("update", test_id, 20, 100, "computing train image mean..."))
            pycaffe.compute_image_mean("lmdb", train_lmdb_path, mean_proto)
            queue.put(("update", test_id, 25, 100, "computing test image mean..."))
            pycaffe.compute_image_mean("lmdb", validation_lmdb_path, mean_proto)

            solver_mode = constant.CAFFE_SOLVER
            if "CAFFE_SOLVER" in os.environ:
                solver_mode = os.environ['CAFFE_SOLVER']

            py_render_template("template/" + template + "/caffenet_train.template", caffe_train_model,
                               mean_file=mean_proto,
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

        queue.put(("log", test_id, "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"))

        queue.put(("log", test_id, "\nStarting to train"))
        queue.put(("update", test_id, 30, 100, "starting to train..."))
        pycaffe.train(caffe_solver, caffe_log, gpu_id, trained_model, ws, queue, test_id, max_iter, snapshot_prefix)

        queue.put(("log", test_id, "\nTrain completed"))

        queue.put(("log", test_id, "\nStarting to test"))

        if not os.path.isfile(ws.workspace("result/slover.prototxt")):
            queue.put(("update", test_id, 90, 100, "starting to test..."))
            py_render_template("template/" + template + "/caffenet_deploy.template", caffe_deploy,
                               num_output=len(classes), img_width=img_width, img_height=img_height)

            set_caffe_gpu(gpu_id)
            queue.put(("log", test_id, "\nReading mean file"))
            queue.put(("update", test_id, 91, 100, "reading mean file..."))
            mean_data = read_mean_data(mean_proto)

            queue.put(("log", test_id, "\nReading neural network model"))
            queue.put(("update", test_id, 92, 100, "reading cnn model..."))
            net = read_model_and_weight(caffe_deploy, snapshot_prefix + "_iter_" + str(max_iter) + ".caffemodel")
            transformer = image_transformers(net, mean_data)

            queue.put(("log", test_id, "Predicting..."))
            queue.put(("update", test_id, 95, 100, "predicting..."))
            prediction = making_predictions(ws.workspace("data/extracted/test"), transformer, net, img_width,
                                            img_height)

            queue.put(("update", test_id, 99, 100, "exporting data..."))
            queue.put(("log", test_id, "Exporting result to csv"))
            export_to_csv(prediction, ws.workspace("result/test_result.csv"))

            queue.put(("log", test_id, "Exporting predict result to folder"))
            export_data(prediction, ws.workspace("data/extracted/test"), ws.workspace("result/data"))

            queue.put(("log", test_id, "Moving log"))
            shutil.copyfile(caffe_log, ws.workspace("result/caffe_train.log"))
            shutil.copyfile(caffe_train_model, ws.workspace("result/model/train_val.prototxt"))
            shutil.copyfile(caffe_deploy, ws.workspace("result/deploy.prototxt"))
            shutil.copyfile(caffe_solver, ws.workspace("result/slover.prototxt"))

            queue.put(("log", test_id, "\nTest completed"))
            queue.put(("done", test_id, "completed"))
    except:
        queue.put(("log", test_id, traceback.format_exc()))
