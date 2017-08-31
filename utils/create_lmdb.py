import glob
import random

import cv2
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from percent_visualize import print_progress
from utils import *
from shutil import copyfile


class CreateLmdb(object):

    def create_lmdb(self, train_path, train_lmdb_path, validation_lmdb_path, classes, test_dir):
        execute('rm -rf  ' + validation_lmdb_path)
        execute('rm -rf  ' + train_lmdb_path)
        train_data = [img for img in glob.glob(train_path + "/*jpg")]

        percent_classes = (70, 20, 10)

        print 'Creating train_lmdb'

        # Shuffle train_data
        random.shuffle(train_data)

        in_db = lmdb.open(train_lmdb_path, map_size=int(1e12))
        total_elements = len(train_data)

        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):
                if self.divide(in_idx, len(train_data), percent_classes) != 0:
                    continue
                self.save_lmdb(in_txn, in_idx, img_path, classes)
                print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
        in_db.close()

        print '\nCreating validation_lmdb'

        in_db = lmdb.open(validation_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):
                if self.divide(in_idx, len(train_data), percent_classes) != 1:
                    continue
                self.save_lmdb(in_txn, in_idx, img_path, classes)
                print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
        in_db.close()

        print '\nCreating test data folder'

        for in_idx, img_path in enumerate(train_data):
            if self.divide(in_idx, len(train_data), percent_classes) != 2:
                continue
            copyfile(img_path, os.path.join(test_dir, os.path.basename(img_path)))
            print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)

        print '\nFinished processing all images'

    def save_lmdb(self, in_txn, in_idx, img_path, classes):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=Constant.IMAGE_WIDTH, img_height=Constant.IMAGE_HEIGHT)
        img_class = os.path.basename(img_path).split(Constant.IMAGE_NAME_SEPARATE_CHARACTER)[0]
        if img_class not in classes:
            label = 1
        else:
            label = 0
        datum = self.make_datum(img, label)
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


    def make_datum(self, img, label):
        # image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=Constant.IMAGE_WIDTH,
            height=Constant.IMAGE_HEIGHT,
            label=label,
            data=np.rollaxis(img, 2).tostring())
