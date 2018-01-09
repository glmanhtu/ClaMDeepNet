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

import numbers
import os
import shutil
import subprocess
from subprocess import call
import cv2

import signal
from mako.template import Template

import pycaffe
from constants import Constant
import re
import Queue

constant = Constant()
caffe = pycaffe
queue = Queue.Queue()
sig_kill = False

def get_queue():
    return queue


def put_log(test_id, message):
    queue.put(("log", test_id, message))


def put_message(message):
    queue.put(message)


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


def execute_command(test_id, command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    # Poll process for new output until finished
    while True:
        if sig_kill:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break
        elif next_line != '':
            put_message(("log", test_id, next_line))
    process.stdout.close()
    process.wait()


def extract_number_iter_from_log(line):
    match = re.findall('Iteration\s*(\d+),\s*Testing', line)
    if len(match) > 0:
        return int(match[0])
    return -1


def execute_train_command(command, test_id, total_iter):
    """

    :type queue: Queue.Queue
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    # Poll process for new output until finished
    while True:
        if sig_kill:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break

        if next_line != '':
            curr_iter = extract_number_iter_from_log(next_line)
            if curr_iter > -1:
                message = "training iter " + str(curr_iter) + "..."
                put_message(("update", test_id, 30 + (curr_iter * 60) / total_iter, 100, message))
                put_message(("log", test_id, next_line))
    process.stdout.close()
    process.wait()


def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def empty_dir(dst):
    for the_file in os.listdir(dst):
        file_path = os.path.join(dst, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def copytree(src, dst, symlinks=False, ignore=None):
    empty_dir(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)