import os
import hashlib
import subprocess
import sys
from mako.template import Template
import cv2
from constants import Constant
import numbers
from subprocess import call
import shutil
import pandas as pd
import matplotlib.pylab as plt
import pycaffe

plt.style.use('ggplot')
constant = Constant()
caffe = pycaffe.Caffe()

def set_workspace(ws):
    ws = os.path.abspath(ws)
    constant.set_workspace(ws)
    if not os.path.isdir(ws):
        os.makedirs(ws)


def workspace(path):
    dir_path = constant.get_workspace() + "/" + path
    file_dir = os.path.dirname(dir_path)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    return dir_path


def template():
    if "CAFFE_TEMPLATE" in os.environ:
        return os.environ['CAFFE_TEMPLATE']
    return constant.CAFFE_TEMPLATE


def gpu_id():
    if "GPU_ID" in os.environ:
        return os.environ['GPU_ID']
    return constant.GPU_ID


def file_already_exists(file_path):
    if os.path.isfile(file_path):
        checksum_file = file_path + ".checksum"
        if os.path.isfile(checksum_file):
            checksum_original = open(checksum_file, 'r').read()
            checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if checksum_original == checksum:
                return True
            os.remove(checksum_file)
        os.remove(file_path)
    return False


def human_2_bytes(s):
    """
    >>> human2bytes('1M')
    1048576
    >>> human2bytes('1G')
    1073741824
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])


def py_render_template(template_file, destination_file, **data):
    template = Template(filename=template_file)
    for parameter in data:
        if not isinstance(data[parameter], numbers.Number):
            if os.path.isdir(data[parameter]) or os.path.isfile(data[parameter]):
                data[parameter] = os.path.abspath(data[parameter])
    result = template.render(**data)
    with open(destination_file, "w") as text_file:
        text_file.write(result)


def save_checksum(file_path):
    read_size = 1024  # You can make this bigger
    checksum1 = hashlib.md5()
    with open(file_path, 'rb') as f:
        data = f.read(read_size)
        while data:
            checksum1.update(data)
            data = f.read(read_size)
    checksum1 = checksum1.hexdigest()
    with open(file_path + ".checksum", "w") as text_file:
        text_file.write(checksum1)


def execute(command):
    call(command, shell=True)


def execute_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break
        sys.stdout.write(next_line)
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


def draw_curve(caffe_log, image_path):
    cwd = os.getcwd()
    os.chdir(os.path.abspath(caffe_log))
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
    title = 'Net Model ' + template()
    if constant.TRAINED_MODEL != "":
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
