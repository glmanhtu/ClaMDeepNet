class Constant(object):

    IMAGE_WIDTH = 200
    IMAGE_HEIGHT = 120

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

    WORKSPACE = "workspace"

    # If you want to fine tune from other model, specific this constant
    TRAINED_MODEL = ""
    # TRAINED_MODEL = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel"
    # TRAINED_MODEL = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"

    def set_workspace(self, workspace):
        self.WORKSPACE = workspace

    def get_workspace(self):
        return self.WORKSPACE
