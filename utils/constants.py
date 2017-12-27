import os


class Constant(object):

    IMAGE_NAME_SEPARATE_CHARACTER = "_"

    # Location of caffe root
    # You also can change this value by set environment CAFFE_ROOT
    CAFFE_ROOT = "/home/ubuntu/caffe"

    # CPU or GPU
    # You also can change this value by set environment CAFFE_SOLVER
    CAFFE_SOLVER = "GPU"

    WORKSPACE = "heobs_sample"

    # If you want to fine tune from other model, specific this constant
    TRAINED_MODEL = ""

    def set_workspace(self, workspace):
        self.WORKSPACE = workspace

    def get_workspace(self):
        if "CAFFE_WORKSPACE" in os.environ:
            return os.path.join("workspace", os.environ['CAFFE_WORKSPACE'])
        return os.path.join("workspace", self.WORKSPACE)
