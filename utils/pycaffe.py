# ClaMDeepNet for classify multiple deep neural network
#
# Copyright (c) 2017 glmanhtu <glmanhtu@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from network.download_file import download_file_strategy, file_already_exists
import utils
import os
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

    def compute_image_mean(self, test_id, backend, lmdb_path, binary_proto_path):
        image_mean_bin = caffe_home() + "/build/tools/compute_image_mean"
        lmdb_path = os.path.abspath(lmdb_path)
        binary_proto_path = os.path.abspath(binary_proto_path)
        command = [image_mean_bin, "-backend=" + backend, lmdb_path, binary_proto_path]
        command = ' '.join(command)
        utils.execute_command(test_id, command)

    def get_curent_train_iter(self, test_id, ws):
        snapshot_path = os.path.abspath(ws.workspace("caffe_model"))
        utils.put_message(("log", test_id, "checking for resume, path: %s" % snapshot_path))
        max_iter = 0

        for root, dirs, files in os.walk(snapshot_path):
            for file in files:
                if file.endswith(".solverstate"):
                    utils.put_message(("log", test_id, "found snapshot: %s" % file))
                    curr_iter = int(re.findall('_(\d+)\.solverstate', file)[0])
                    if max_iter < curr_iter:
                        max_iter = curr_iter

        utils.put_message(("log", test_id, "current iter: %d" % max_iter))
        return max_iter

    def train(self, solver, log, gpu_id, trained_model, ws, test_id, total_iter):
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = caffe_home() + "/build/tools/caffe"
        resume_sloverstate = self.get_curent_train_iter(test_id, ws)
        command = ["GLOG_minloglevel=0", caffe_bin, "train", "--solver=" + solver]

        if trained_model != "":
            trained_model_path = ws.workspace("trained_models/trained_model.caffemodel")
            if not file_already_exists(trained_model_path):
                download_file_strategy(trained_model, trained_model_path)
            command.extend(["--weights", trained_model_path])
        if resume_sloverstate == total_iter:
            return
        if Constant.CAFFE_SOLVER == "GPU":
            command.extend(["-gpu=" + gpu_id])

        command.extend(["2>&1 | tee", log])
        utils.put_message(("log", test_id, "Train command: %s" % ' '.join(command)))

        utils.execute_train_command(' '.join(command), test_id, total_iter)

