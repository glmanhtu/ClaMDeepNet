import csv
import os
import shutil
import sys
import threading
import Queue

from heobs_image_classification import heobs_image_classification
from utils.multiple_level_progress import MultipleLevelProgress
from utils.workspace import Workspace


def read_test_config(test_config):
    tests = []
    with open(test_config, "r") as ins:
        csv_reader = csv.reader(ins, delimiter=',', quotechar='"')
        next(csv_reader)
        for row in csv_reader:
            tests.append({
                'arch': row[0],
                'lr': float(row[1]),
                'stepsize': int(row[2]),
                'img_width': int(row[3]),
                'img_height': int(row[4]),
                'train_batch_size': int(row[5]),
                'test_batch_size': int(row[6]),
                'gpu_id': row[7],
                'finetune': row[8],
                'parallel': row[9],
                'max_iter': int(row[10])
            })
    return tests


def get_parallel(test_list):
    parallel_list = []
    for test_info in test_list:
        if test_info['parallel'] not in parallel_list:
            parallel_list.append(test_info['parallel'])
    return parallel_list


def generate_workspace(test_info):
    workspace_parts = [test_info['arch'],
                       str(test_info['max_iter']),
                       str(test_info['img_width']),
                       str(test_info['img_height']),
                       test_info['gpu_id'],
                       str(test_info['lr']),
                       str(test_info['stepsize']),
                       str(test_info['train_batch_size']),
                       str(test_info['test_batch_size'])]
    if test_info['finetune'] != "":
        workspace_parts.append("finetune")

    return os.path.join("workspace", '_'.join(workspace_parts))


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def reporter(q, nworkers):
    multiple_level_progress = MultipleLevelProgress(nworkers, 8)
    while nworkers > 0:
        msg = q.get()
        if msg[0] == "update":
            test_id, current, total, message = msg[1:]
            multiple_level_progress.update(test_id, current, message)
        elif msg[0] == "done":
            nworkers = nworkers - 1


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


if __name__ == '__main__':
    test_config = sys.argv[1]
    if os.path.isfile(test_config):
        queue = Queue.Queue()
        tests = read_test_config(test_config)
        parallels = get_parallel(tests)
        test_id = 0
        monitor = threading.Thread(target=reporter, args=(queue, len(tests)))
        monitor.start()
        for idx, parallel in enumerate(parallels):
            threads = []
            workspaces = []
            for test in tests:
                if test['parallel'] == parallel:
                    workspace = Workspace(generate_workspace(test))
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
                        queue,
                        test_id
                    ])
                    thread.start()
                    threads.append(thread)
                    test_id += 1
            for thread in threads:
                thread.join()
            for test in tests:
                if test['parallel'] == parallel:
                    workspace = Workspace(generate_workspace(test))
                    collect_result(workspace, test)
        monitor.join()

    else:
        raise Exception("Please specific test config file")
