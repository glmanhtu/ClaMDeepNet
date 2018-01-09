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

import glob
import random
from shutil import copyfile

import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from utils import *


class CreateLmdb(object):
    def create_lmdb(self, train_path, train_lmdb_path, validation_lmdb_path, classes, test_path, img_width, img_height):
        empty_dir(validation_lmdb_path)
        empty_dir(train_lmdb_path)
        empty_dir(test_path)
        train_data = [img for img in glob.glob(train_path + "/*jpg")]

        # Shuffle train_data
        random.shuffle(train_data)

        img_list = {}

        for img in train_data:
            for clazz in classes:
                if clazz not in img_list:
                    img_list[clazz] = []
                if clazz in img:
                    img_list[clazz].append(img)
                    break

        percent_classes = Constant.IMAGE_TRAIN_VAL_TEST

        in_db = lmdb.open(train_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            img_train = []
            for clazz in img_list:
                total_elements = len(img_list[clazz])
                for in_idx, img_path in enumerate(img_list[clazz]):
                    if self.divide(in_idx, total_elements, percent_classes) != 0:
                        continue
                    img_train.append(img_path)
            random.shuffle(img_train)
            for idx, img in enumerate(img_train):
                self.save_lmdb(in_txn, idx, img, classes, img_width, img_height)

        in_db.close()

        in_db = lmdb.open(validation_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            img_val = []
            for clazz in img_list:
                total_elements = len(img_list[clazz])
                for in_idx, img_path in enumerate(img_list[clazz]):
                    if self.divide(in_idx, total_elements, percent_classes) != 1:
                        continue
                    img_val.append(img_path)
            random.shuffle(img_val)
            for idx, img in enumerate(img_val):
                self.save_lmdb(in_txn, idx, img, classes, img_width, img_height)

        in_db.close()

        os.makedirs(test_path)
        test_img_count = 0
        for clazz in img_list:
            total_elements = len(img_list[clazz])
            for in_idx, img_path in enumerate(img_list[clazz]):
                if self.divide(in_idx, total_elements, percent_classes) != 2:
                    continue
                copyfile(img_path, os.path.join(test_path, os.path.basename(img_path)))
                test_img_count += 1

    def save_lmdb(self, in_txn, in_idx, img_path, classes, img_width, img_height):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=img_width, img_height=img_height)
        img_class = os.path.basename(img_path).split(Constant.IMAGE_NAME_SEPARATE_CHARACTER)[0]
        label = classes.index(img_class)
        datum = self.make_datum(img, label, img_width, img_height)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())

    def divide(self, index, total, percent_classes):
        actual_percent = []
        for idx, percent in enumerate(percent_classes):
            actual_percent.append(percent)
            if idx > 0:
                actual_percent[idx] += actual_percent[idx - 1]
        for idx, percent in enumerate(actual_percent):
            if index / float(total) <= percent / 100.0:
                return idx

    def make_datum(self, img, label, img_width, img_height):
        # image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=img_width,
            height=img_height,
            label=label,
            data=np.rollaxis(img, 2).tostring())
