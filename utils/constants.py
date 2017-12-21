import os

from network.google_file import GoogleFile


class Constant(object):

    IMAGE_WIDTH = 227
    IMAGE_HEIGHT = 227

    CAFFE_TEMPLATE = "alexnet"

    IMAGE_NAME_SEPARATE_CHARACTER = "_"

    NUMBER_OUTPUT = 3

    # Location of caffe root
    # You also can change this value by set environment CAFFE_ROOT
    CAFFE_ROOT = "/home/ubuntu/caffe"

    # CPU or GPU
    # You also can change this value by set environment CAFFE_SOLVER
    CAFFE_SOLVER = "GPU"

    GPU_ID = "0"

    WORKSPACE = "heobs_sample"

    # If you want to fine tune from other model, specific this constant
    TRAINED_MODEL = ""
    GG_TRAINED_MODEL = ""
    # TRAINED_MODEL = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
    # TRAINED_MODEL = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel"
    # TRAINED_MODEL = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"
    # TRAINED_MODEL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel"
    # GG_TRAINED_MODEL = GoogleFile('1OqFQWTzYYFjT-J9N3zoQ8UtGC8M4LfxB', 'ResNet-152-model.caffemodel', '/tmp/ResNet-152-model.caffemodel')

    def set_workspace(self, workspace):
        self.WORKSPACE = workspace

    def get_workspace(self):
        if "CAFFE_WORKSPACE" in os.environ:
            return os.path.join("workspace", os.environ['CAFFE_WORKSPACE'])
        return os.path.join("workspace", self.WORKSPACE)
