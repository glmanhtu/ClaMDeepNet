from network.download_file import download_file_strategy, file_already_exists
import utils
import os
from os import listdir
from os.path import isfile, join
from constants import Constant
import re


def caffe_home():
    if "CAFFE_ROOT" in os.environ:
        return os.environ['CAFFE_ROOT']
    return Constant.CAFFE_ROOT


def check_caffe_root():
    if os.path.isfile(caffe_home() + "/build/tools/caffe"):
        return True
    return False

class PyCaffe(object):

    def compute_image_mean(self, backend, lmdb_path, binary_proto_path, logger):
        image_mean_bin = caffe_home() + "/build/tools/compute_image_mean"
        lmdb_path = os.path.abspath(lmdb_path)
        binary_proto_path = os.path.abspath(binary_proto_path)
        command = [image_mean_bin, "-backend=" + backend, lmdb_path, binary_proto_path]
        command = ' '.join(command)
        utils.execute_command(command, logger)

    def get_resume_sloverstate(self, ws, snapshot_prefix):
        snapshot_path = os.path.dirname(snapshot_prefix)
        max_iter = 0
        sloverstate = ""
        for file in listdir(snapshot_path):
            file_path = join(snapshot_path, file)
            if isfile(file_path):
                if "sloverstate" in file_path:
                    curr_iter = int(re.findall('snapshot_iter_(\d+)\.solverstate', file_path)[0])
                    if max_iter < curr_iter:
                        max_iter = curr_iter
                        sloverstate = file_path
        return sloverstate

    def train(self, solver, log, gpu_id, trained_model, ws, logger, queue, test_id, total_iter, snapshot_prefix):
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = caffe_home() + "/build/tools/caffe"
        resume_sloverstate = self.get_resume_sloverstate(ws, snapshot_prefix)
        command = ["GLOG_minloglevel=0", caffe_bin, "train", "--solver=" + solver]

        if trained_model != "":
            trained_model_path = ws.workspace("trained_models/trained_model.caffemodel")
            if not file_already_exists(trained_model_path):
                download_file_strategy(trained_model, trained_model_path)
            command.extend(["--weights", trained_model_path])
        if resume_sloverstate != "":
            if str(total_iter) + ".sloverstate" in resume_sloverstate:
                return
            command.extend(["--snapshot", resume_sloverstate])
        if Constant.CAFFE_SOLVER == "GPU":
            command.extend(["-gpu=" + gpu_id])

        command.extend(["2>&1 | tee -a", log])
        utils.execute_train_command(' '.join(command), logger, queue, test_id, total_iter)

