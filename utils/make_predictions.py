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

import errno
import glob
from shutil import copyfile
import os
import caffe
import cv2
import numpy as np
from caffe.proto import caffe_pb2

from constants import Constant
from utils import transform_img

def set_caffe_gpu(gpu_id):
    """

    :type gpu_id: str
    """
    caffe.set_mode_gpu()
    if "," in gpu_id:
        caffe.set_device(int(gpu_id.split(',')[0]))
    else:
        caffe.set_device(int(gpu_id))

def read_mean_data(mean_file):

    mean_blob = caffe_pb2.BlobProto()
    with open(mean_file) as f:
        mean_blob.ParseFromString(f.read())
    return np.asarray(mean_blob.data, dtype=np.float32)\
        .reshape(mean_blob.channels, mean_blob.height, mean_blob.width)


def read_model_and_weight(model_deploy_file, model_weight_file):
    return caffe.Net(model_deploy_file, caffe.TEST, weights=model_weight_file)


def image_transformers(net, mean_data):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    if mean_data is not None:
        transformer.set_mean('data', mean_data)
    transformer.set_transpose('data', (2, 0, 1))
    return transformer


def making_predictions(test_img_path, transformer, net, img_width, img_height):
    test_img_paths = [img_path for img_path in glob.glob(test_img_path + "/*jpg")]

    test_ids = []
    predictions = []
    for img_path in test_img_paths:
        pred_probas = single_making_prediction(img_path, transformer, net, img_width, img_height)

        test_ids = test_ids + [img_path.split('/')[-1][:-4]]
        if pred_probas[0][pred_probas.argmax()] < 0.5:
            predictions = predictions + [len(pred_probas[0])]
        else:
            predictions = predictions + [pred_probas.argmax()]
    return [test_ids, predictions]


def single_making_prediction(img_path, transformer, net, img_width, img_height):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=img_width, img_height=img_height)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    return out['prob']


def export_data(prediction, dataset_dir, export_dir):
    for i in range(len(prediction[0])):
        destination_file = os.path.join(export_dir, str(prediction[1][i]), str(prediction[0][i]) + ".jpg")
        if not os.path.exists(os.path.dirname(destination_file)):
            try:
                os.makedirs(os.path.dirname(destination_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        source_file = os.path.join(dataset_dir, str(prediction[0][i]) + ".jpg")
        copyfile(source_file, destination_file)


def show_result(classes, prediction):
    result = []
    for i in range(len(classes)):
        result.append([])
        for j in range(len(classes) + 1):
            result[i].append(0)
    for i in range(len(prediction[0])):
        actual = classes.index(prediction[0][i].split(Constant.IMAGE_NAME_SEPARATE_CHARACTER)[0])
        predict = prediction[1][i]
        result[actual][predict] += 1
    print result


def export_to_csv(prediction, export_file):
    with open(export_file, "w") as f:
        f.write("id,label\n")
        for i in range(len(prediction[0])):
            f.write(str(prediction[0][i]) + "," + str(prediction[1][i]) + "\n")
    f.close()
