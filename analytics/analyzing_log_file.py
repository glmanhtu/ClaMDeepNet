import os
import re
import sys
import subprocess

class AnalyzingLogFile:

    log_test = 'train.log.test'
    log_train = 'train.log.train'

    def __init__(self):
        self.execute_command("chmod +x parse_log.sh")
        self.execute_command("chmod +x extract_seconds.py")
        pass

    def __read_train_result_file(self):
        if os.path.isfile(self.log_train):
            result = []
            with open(self.log_train, "r") as ins:
                for idx, line in enumerate(ins):
                    if idx == 0:
                        continue
                    content = re.compile("\s+").split(line.strip())
                    result.append({'iter': content[0],
                                   'second': content[1], 'train_loss': content[2]})
            return result
        raise Exception('train result not found')

    def __read_test_result_file(self):
        if os.path.isfile(self.log_test):
            result = []
            with open(self.log_test, "r") as ins:
                for idx, line in enumerate(ins):
                    if idx == 0:
                        continue
                    content = re.compile("\s+").split(line.strip())
                    result.append({'iter': content[0],
                                   'second': content[1], 'test_accuracy': content[2], 'test_loss': content[3]})
            return result
        raise Exception('train result not found')

    def __calculate_overfitting(self, train_result, test_result):

        total_distance = 0
        total = 0
        for itx, test in enumerate(test_result):
            for train in train_result:
                if test['iter'] == train['iter']:
                    distance = abs(float(test['test_loss']) - float(train['train_loss']))
                    total_distance += distance * itx
                    total += itx
        return total_distance / total

    def __calculate_total_time(self, train_result, test_result):
        return float(train_result[len(train_result) - 1]['second'])

    def __calculate_max_accuracy(self, test_result):
        max_acc = 0
        for test in test_result:
            if max_acc < float(test['test_accuracy']):
                max_acc = float(test['test_accuracy'])
        return max_acc

    def execute_command(self, command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Poll process for new output until finished
        while True:
            next_line = process.stdout.readline()
            if next_line == '' and process.poll() is not None:
                break
            sys.stdout.write(next_line)
            sys.stdout.flush()

    def analyzing(self, log_file_path):
        self.execute_command("./parse_log.sh %s" % log_file_path)

        train_result = self.__read_train_result_file()
        test_result = self.__read_test_result_file()

        over_fitting = self.__calculate_overfitting(train_result, test_result)
        total_time = self.__calculate_total_time(train_result, test_result)
        max_accuracy = self.__calculate_max_accuracy(test_result)
        accuracy_at_max_iter = float(test_result[len(test_result) - 1]['test_accuracy'])
        test_loss_at_max_iter = float(test_result[len(test_result) - 1]['test_loss'])
        train_loss_at_max_iter = float(train_result[len(train_result) - 1]['train_loss'])

        os.unlink(self.log_test)
        os.unlink(self.log_train)
        
        return {
            "over_fitting": over_fitting,
            "total_time": total_time,
            "max_accuracy": max_accuracy,
            "accuracy_at_max_iter": accuracy_at_max_iter,
            "test_loss_at_max_iter": test_loss_at_max_iter,
            "train_loss_at_max_iter": train_loss_at_max_iter
        }