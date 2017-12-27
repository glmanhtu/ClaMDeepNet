import glob
import random
from shutil import copyfile

import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from percent_visualize import print_progress
from utils import *


class CreateLmdb(object):

    def create_lmdb(self, train_path, train_lmdb_path, validation_lmdb_path, classes, test_path):
        execute('rm -rf  ' + validation_lmdb_path)
        execute('rm -rf  ' + train_lmdb_path)
        execute('rm -rf  ' + test_path)
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

        percent_classes = (80, 10, 10)

        print 'Creating train_lmdb'

        in_db = lmdb.open(train_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            img_train = []
            for clazz in img_list:
                total_elements = len(img_list[clazz])
                for in_idx, img_path in enumerate(img_list[clazz]):
                    print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
                    if self.divide(in_idx, total_elements, percent_classes) != 0:
                        continue
                    img_train.append(img_path)
            random.shuffle(img_train)
            for idx, img in enumerate(img_train):
                self.save_lmdb(in_txn, idx, img, classes)

            print '\n Total images for train: ' + str(len(img_train))
        in_db.close()

        in_db = lmdb.open(validation_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            img_val = []
            for clazz in img_list:
                total_elements = len(img_list[clazz])
                for in_idx, img_path in enumerate(img_list[clazz]):
                    print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
                    if self.divide(in_idx, total_elements, percent_classes) != 1:
                        continue
                    img_val.append(img_path)
            random.shuffle(img_val)
            for idx, img in enumerate(img_val):
                self.save_lmdb(in_txn, idx, img, classes)

            print '\n Total images for validate: ' + str(len(img_val))
        in_db.close()

        os.makedirs(test_path)
        test_img_count = 0
        for clazz in img_list:
            total_elements = len(img_list[clazz])
            for in_idx, img_path in enumerate(img_list[clazz]):
                print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
                if self.divide(in_idx, total_elements, percent_classes) != 2:
                    continue
                copyfile(img_path, os.path.join(test_path, os.path.basename(img_path)))
                test_img_count += 1

        print '\n total images for test: ' + str(test_img_count)
        print '\nFinished processing all images'

    def save_lmdb(self, in_txn, in_idx, img_path, classes):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=Constant.IMAGE_WIDTH, img_height=Constant.IMAGE_HEIGHT)
        img_class = os.path.basename(img_path).split(Constant.IMAGE_NAME_SEPARATE_CHARACTER)[0]
        label = classes.index(img_class)
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
