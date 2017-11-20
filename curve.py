#!flask/bin/python
import matplotlib
matplotlib.use('Agg')
import random
import string

from datetime import timedelta
from functools import update_wrapper

from flask import Flask, request, jsonify, current_app, make_response, render_template, send_file, send_from_directory

from utils.make_predictions import *
import sys
import subprocess
import pandas as pd

import matplotlib.pylab as plt
from utils import pycaffe
import shutil
from utils import zip_utils
import threading
import logging

plt.style.use('ggplot')


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

set_workspace(os.path.join("workspace", "heobs_sample"))
caffe_log = os.path.realpath(workspace("caffe_model/caffe_train.log"))
image_path = os.path.realpath(workspace("caffe_model/caffe_curve.png"))
result_path = os.path.realpath(workspace("result/test_result.csv"))
result_data_path = os.path.realpath(workspace("result/data"))
caffe_model = os.path.realpath(workspace("caffe_model/caffenet_train.prototxt"))
caffe_model_img = os.path.realpath(workspace("caffe_model/caffenet_train.png"))
caffe = pycaffe.Caffe()
app = Flask(__name__, template_folder="web/template")

app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
model_log_dir_path = os.path.realpath(os.path.dirname(caffe_log))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('imageview.html')


@app.route('/result.csv', methods=['GET'])
@crossdomain(origin='*')
def get_result():
    file_dir = os.path.dirname(result_path)
    file_name = os.path.basename(result_path)
    return send_from_directory(file_dir, file_name)


@app.route('/data/result.zip', methods=['GET'])
@crossdomain(origin='*')
def get_data_result():
    tmp_archive = zip_utils.zip_path(result_data_path)
    file_dir = os.path.dirname(tmp_archive)
    file_name = os.path.basename(tmp_archive)
    return send_from_directory(file_dir, file_name)


@app.route('/model', methods=['GET'])
@crossdomain(origin='*')
def get_model():
    command = caffe.caffe_home() + '/python/draw_net.py ' + caffe_model + ' ' + caffe_model_img
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return send_file(caffe_model_img, mimetype='image/png')

@app.route('/curve', methods = ['GET'])
@crossdomain(origin='*')
def curve():
    # Get directory where the model logs is saved, and move to it

    os.chdir(model_log_dir_path)

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
    plt.figure(figsize=(10, 6), dpi=80)
    fig, ax1 = plt.subplots()

    # Plotting training and test losses
    train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red', alpha=.5)
    test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=1, color='green')
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.tick_params(labelsize=12)
    # Plotting test accuracy
    ax2 = ax1.twinx()
    test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'], linewidth=1, color='blue')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.tick_params(labelsize=12)
    # Adding legend
    plt.legend([train_loss, test_loss, test_accuracy], ['Training Loss', 'Test Loss', 'Test Accuracy'],
               bbox_to_anchor=(1, 0.8))
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
    return send_file(image_path, mimetype='image/png')


def run_app():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8081, debug=False)


def run_thread_app():
    t = threading.Thread(target=run_app)
    t.setDaemon(True)
    t.start()


if __name__ == '__main__':
    run_app()
