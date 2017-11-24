from utils import utils
import os
import re


class AnalyzingLogFile:

    log_test = 'train.log.test'
    log_train = 'train.log.train'

    def __init__(self):
        utils.execute_command("chmod +x parse_log.sh")
        utils.execute_command("chmod +x extract_seconds.py")
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

    def analyzing(self, log_file_path):
        utils.execute_command("./parse_log.sh %s" % log_file_path)

        train_result = self.__read_train_result_file()
        test_result = self.__read_test_result_file()

        over_fitting = self.__calculate_overfitting(train_result, test_result)
        total_time = self.__calculate_total_time(train_result, test_result)
        max_accurency = self.__calculate_max_accuracy(test_result)
        accurency_at_max_iter = float(test_result[len(test_result) - 1]['test_accuracy'])
        test_loss_at_max_iter = float(test_result[len(test_result) - 1]['test_loss'])
        train_loss_at_max_iter = float(train_result[len(train_result) - 1]['train_loss'])

        os.unlink(self.log_test)
        os.unlink(self.log_train)
        
        return {
            "over_fitting": over_fitting,
            "total_time": total_time,
            "max_accuracy": max_accurency,
            
        }

analyzing = AnalyzingLogFile()
analyzing.analyzing("/home/glmanhtu/Documents/test-result/20k-iterators-2/alexnet/0.001lr/finetune/train.log")