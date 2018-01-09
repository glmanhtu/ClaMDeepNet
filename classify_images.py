import csv
import logging
import os
import traceback
import shutil
import sys
import threading

import signal

import time

os.environ['GLOG_minloglevel'] = '2'
from heobs_image_classification import heobs_image_classification
from utils import pycaffe
from utils.multiple_level_progress import MultipleLevelProgress
from utils.workspace import Workspace
from utils import utils

logger = logging.getLogger(__name__)
hdlr = logging.FileHandler("debug.log")
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def read_test_config(test_config):
    tests = []
    with open(test_config, "r") as ins:
        csv_reader = csv.reader(ins, delimiter=',', quotechar='"')
        next(csv_reader)
        for row in csv_reader:
            tests.append({
                'test_id': int(row[0]),
                'arch': row[1],
                'lr': float(row[2]),
                'stepsize': int(row[3]),
                'img_width': int(row[4]),
                'img_height': int(row[5]),
                'train_batch_size': int(row[6]),
                'test_batch_size': int(row[7]),
                'gpu_id': row[8],
                'finetune': row[9],
                'parallel': row[10],
                'max_iter': int(row[11])
            })
    return tests


def get_parallel(test_list):
    parallel_list = []
    for test_info in test_list:
        if test_info['parallel'] not in parallel_list:
            parallel_list.append(test_info['parallel'])
    return parallel_list


def generate_workspace(test_id):
    return os.path.join("workspace", str(test_id))


def copytree(src, dst, symlinks=False, ignore=None):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def reporter(q, nworkers):
    multiple_level_progress = MultipleLevelProgress(nworkers, 100)
    while nworkers > 0:
        if utils.sig_kill:
            return
        msg = q.get()
        if msg[0] == "update":
            test_id, current, total, message = msg[1:]
            multiple_level_progress.update(test_id, current, message)
        elif msg[0] == "done":
            test_id, message = msg[1:]
            multiple_level_progress.update(test_id, 100, message)
            nworkers = nworkers - 1
        else:
            test_id, log_message = msg[1:]
            logger.debug("[test:%d] %s" % (test_id, log_message))


def collect_result(test_space, test_info):
    """

    :param test_info: object
    :type test_space: Workspace
    """
    result = test_space.workspace("result")
    destination = os.path.join("result",
                               test_info['arch'],
                               str(test_info['lr']) + 'lr',
                               "stepsize-" + str(test_info['stepsize']),
                               "train_batch_size-" + str(test_info['train_batch_size']),
                               "test_batch_size-" + str(test_info['test_batch_size'])
                               )
    if test_info['finetune'] != "":
        destination = os.path.join(destination, "finetune")
    else:
        destination = os.path.join(destination, "conventional")

    if not os.path.isdir(destination):
        os.makedirs(destination)
    copytree(result, destination)


def signal_term_handler(signal, frame):
    utils.sig_kill = True
    logger.error("Killed")

if __name__ == '__main__':

    test_config = sys.argv[1]
    signal.signal(signal.SIGTERM, signal_term_handler)
    if os.path.isfile(test_config):
        tests = read_test_config(test_config)
        parallels = get_parallel(tests)

        if not pycaffe.check_caffe_root():
            print "Caffe root incorrect"
            print "Please specific caffe root directory by command: "
            print "\texport CAFFE_ROOT=/path/to/caffe"
            sys.exit()

        print "Starting..."
        print "Total tests: %d" % len(tests)
        print "Total parallel: %d" % len(parallels)

        monitor = threading.Thread(target=reporter, args=(utils.get_queue(), len(tests)))
        monitor.setDaemon(True)
        monitor.start()
        try:
            for idx, parallel in enumerate(parallels):
                threads = []
                workspaces = []
                for test in tests:
                    if test['parallel'] == parallel:
                        workspace = Workspace(generate_workspace(test['test_id']))
                        workspaces.append(workspace)
                        thread = threading.Thread(target=heobs_image_classification, args=[
                            test['arch'],
                            test['max_iter'],
                            test['img_width'],
                            test['img_height'],
                            test['gpu_id'],
                            test['lr'],
                            test['stepsize'],
                            test['train_batch_size'],
                            test['test_batch_size'],
                            test['finetune'],
                            workspace,
                            test['test_id']
                        ])
                        thread.start()
                        threads.append((thread, test))
                for thread, test in threads:
                    thread.join()

                for thread, test in threads:
                    workspace = Workspace(generate_workspace(test['test_id']))
                    collect_result(workspace, test)

        except KeyboardInterrupt:
            utils.sig_kill = True
            logger.error("User exit")
        except:
            utils.sig_kill = True
            logger.error(traceback.format_exc())

    else:
        raise Exception("Please specific test config file")
