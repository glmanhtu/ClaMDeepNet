from network.download_file import download_file_strategy, file_already_exists
from utils import *


def caffe_home():
    if "CAFFE_ROOT" in os.environ:
        return os.environ['CAFFE_ROOT']
    return constant.CAFFE_ROOT


class PyCaffe(object):

    def compute_image_mean(self, backend, lmdb_path, binary_proto_path, logger):
        image_mean_bin = caffe_home() + "/build/tools/compute_image_mean"
        lmdb_path = os.path.abspath(lmdb_path)
        binary_proto_path = os.path.abspath(binary_proto_path)
        command = [image_mean_bin, "-backend=" + backend, lmdb_path, binary_proto_path]
        command = ' '.join(command)
        execute_command(command, logger)

    def train(self, solver, log, gpu_id, trained_model, ws, logger):
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = caffe_home() + "/build/tools/caffe"

        command = [caffe_bin, "train", "--solver=" + solver]

        if trained_model != "":
            trained_model_path = os.path.join(ws.workspace("trained_models"), "trained_model.caffemodel")
            if not file_already_exists(trained_model_path):
                download_file_strategy(trained_model, trained_model_path)
            command.extend(["--weights", trained_model_path])
        if constant.CAFFE_SOLVER == "GPU":
            command.extend(["-gpu=" + gpu_id])

        command.extend(["2>&1 | tee", log])
        execute_command(' '.join(command), logger)

