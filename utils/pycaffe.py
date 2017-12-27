from network.download_file import download_file_strategy, file_already_exists
from utils import *


def caffe_home():
    if "CAFFE_ROOT" in os.environ:
        return os.environ['CAFFE_ROOT']
    return constant.CAFFE_ROOT


class Caffe(object):

    def compute_image_mean(self, backend, lmdb_path, binary_proto_path):
        print ("Computing image mean from %s" % lmdb_path)
        image_mean_bin = caffe_home() + "/build/tools/compute_image_mean"
        lmdb_path = os.path.abspath(lmdb_path)
        binary_proto_path = os.path.abspath(binary_proto_path)
        command = [image_mean_bin, "-backend=" + backend, lmdb_path, binary_proto_path]
        command = ' '.join(command)
        execute(command)
        print ("Completed")

    def train(self, solver, log):
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = caffe_home() + "/build/tools/caffe"

        command = [caffe_bin, "train", "--solver=" + solver]

        if constant.TRAINED_MODEL != "":
            trained_model_path = os.path.join(workspace("trained_models"), "trained_model.caffemodel")
            if not file_already_exists(trained_model_path):
                print "Downloading trained model"
                download_file_strategy(constant.TRAINED_MODEL, trained_model_path)
                print "Downloaded model"
            else:
                print "trained model already downloaded"
            command.extend(["--weights", trained_model_path])
        if constant.CAFFE_SOLVER == "GPU":
            command.extend(["-gpu=" + gpu_id()])

        command.extend(["2>&1 | tee", log])

        print "\nRun command: \n\n"
        print command
        print "\n\n\n"
        execute_command(' '.join(command))

