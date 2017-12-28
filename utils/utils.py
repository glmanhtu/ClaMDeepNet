import numbers
import os
import shutil
import subprocess
import sys
from subprocess import call
import cv2
from mako.template import Template

import pycaffe
from constants import Constant
import re

constant = Constant()
caffe = pycaffe


def py_render_template(template_file, destination_file, **data):
    template = Template(filename=template_file)
    for parameter in data:
        if not isinstance(data[parameter], numbers.Number):
            if os.path.isdir(data[parameter]) or os.path.isfile(data[parameter]):
                data[parameter] = os.path.abspath(data[parameter])
    result = template.render(**data)
    with open(destination_file, "w") as text_file:
        text_file.write(result)


def execute(command):
    call(command, shell=True)


def execute_command(command, logger):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break
        if next_line != '':
            logger.debug(next_line)
        sys.stdout.flush()


def extract_number_iter_from_log(line):
    match = re.findall('Iteration\s*(\d+),\s*Testing', line)
    if len(match) > 0:
        return int(match[0])
    return -1


def execute_train_command(command, logger, queue, test_id, total_iter):
    """

    :type logger: Logger
    :type queue: Queue.Queue
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break

        if next_line != '':
            curr_iter = extract_number_iter_from_log(next_line)
            if curr_iter > -1:
                queue.put(("update", test_id, 30 + (curr_iter * 60) / total_iter, 100,
                           "training iter " + str(curr_iter) + "..."))
            logger.debug(next_line)
        sys.stdout.flush()


def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def empty_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
