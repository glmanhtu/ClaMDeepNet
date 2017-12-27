import numbers
import os
import shutil
import subprocess
import sys
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pylab as plt
import pandas as pd
from mako.template import Template

import pycaffe
from constants import Constant

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
        logger.debug(next_line)


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


def draw_curve(caffe_log, image_path, template, trained_model):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(caffe_log))
    '''
        Generating training and test logs
        '''
    # Parsing training/validation logs
    command = caffe.caffe_home() + '/tools/extra/parse_log.sh ' + caffe_log
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    # Read training and test logs
    train_log_path = caffe_log + '.train'
    test_log_path = caffe_log + '.test'
    train_log = pd.read_csv(train_log_path, delim_whitespace=True)
    test_log = pd.read_csv(test_log_path, delim_whitespace=True)

    '''
    Making learning curve
    '''
    plt.figure(figsize=(12, 6), dpi=120)
    fig, ax1 = plt.subplots()

    # Plotting training and test losses
    train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red', alpha=.5)
    test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=1, color='green')
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.tick_params(labelsize=11)
    # Plotting test accuracy
    ax2 = ax1.twinx()
    test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'], linewidth=1, color='blue')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.tick_params(labelsize=11)
    # Adding legend
    plt.legend([train_loss, test_loss, test_accuracy], ['Training Loss', 'Test Loss', 'Test Accuracy'],
               bbox_to_anchor=(1, 0.9))
    title = 'Net Model ' + template
    if trained_model != "":
        title += " - Finetune"
    plt.title(title, fontsize=16)
    # Saving learning curve
    plt.savefig(image_path)

    '''
    Deleting training and test logs
    '''
    os.remove(train_log_path)
    os.remove(test_log_path)
    os.chdir(cwd)
