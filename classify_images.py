import sys
import threading
import os
import shutil

from heobs_image_classification import heobs_image_classification
from utils.workspace import Workspace


def read_test_config(test_config):
    tests = []
    with open(test_config, "r") as ins:
        for idx, line in enumerate(ins):
            if idx == 0:
                continue
            content = line.strip().split(",")
            tests.append({
                'arch': content[0],
                'lr': float(content[1]),
                'stepsize': int(content[2]),
                'img_width': int(content[3]),
                'img_height': int(content[4]),
                'train_batch_size': int(content[5]),
                'test_batch_size': int(content[6]),
                'gpu_id': content[7],
                'finetune': content[8],
                'parallel': content[9],
                'max_iter': int(content[10])
            })
    return tests


def get_parallel(test_list):
    parallel_list = []
    for test_info in test_list:
        if test_info['parallel'] not in parallel_list:
            parallel_list.append(test_info['parallel'])
    return parallel_list


def generate_workspace(test_info):
    workspace = [test_info['arch'],
                 test_info['max_iter'],
                 test_info['img_width'],
                 test_info['img_height'],
                 test_info['gpu_id'],
                 test_info['lr'],
                 test_info['stepsize'],
                 test_info['train_batch_size'],
                 test_info['test_batch_size']]
    if test_info['finetune'] != "":
        workspace.append("finetune")

    return os.path.join("workspace", '_'.join(workspace))


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def collect_result(test_space, test_info):
    """

    :param test_info: object
    :type test_space: Workspace
    """
    result = test_space.workspace("result")
    destination = os.path.join("result",
                               test_info['arch'],
                               test_info['lr'] + 'lr',
                               "stepsize-" + test_info['stepsize'],
                               "train_batch_size-" + test_info['train_batch_size'],
                               "test_batch_size-" + test_info['test_batch_size']
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
        tests = read_test_config(test_config)
        parallels = get_parallel(tests)
        for parallel in parallels:
            threads = []
            workspaces = []
            for test in tests:
                if test['parallel'] == parallel:
                    print "Starting test: ", test
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
                        workspace
                    ])
                    thread.start()
                    threads.append(thread)
            for thread in threads:
                thread.join()
            for test in tests:
                if test['parallel'] == parallel:
                    workspace = Workspace(generate_workspace(test))
                    collect_result(workspace, test)

    else:
        raise Exception("Please specific test config file")
