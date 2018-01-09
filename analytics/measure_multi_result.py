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

import os
import sys
from analyzing_log_file import AnalyzingLogFile
from analyzing_test_result_file import AnalyzingTestResult

analyzing_log_file = AnalyzingLogFile()
analyzing_test_file = AnalyzingTestResult()
classes = ["being", "heritage", "scenery"]

def processing_log_dir(result_dir, log_dir):
    relation_path = log_dir.replace(result_dir + "/", '')
    options = relation_path.split("/")
    log_file = os.path.join(log_dir, "caffe_train.log")
    result_file = os.path.join(log_dir, "test_result.csv")

    log_file_result = analyzing_log_file.analyzing(log_file)
    test_file_result = analyzing_test_file.analyzing(result_file, classes)

    return {"log_file_result": log_file_result, "test_file_result": test_file_result, "options": options}


if __name__ == '__main__':
    result_dir = sys.argv[1]
    if os.path.isdir(result_dir):
        result_dir = os.path.abspath(result_dir)

        report_column = ["test_name", "test_accuracy", "train_loss_at_max_iter", "val_loss_at_max_iter"]
        report_column.extend(["accuracy_at_max_iter", "max_accuracy", "over_fitting", "total_time"])

        report_content = ",".join(report_column) + "\n"

        for root, dirs, files in os.walk(result_dir):
            for f in dirs:
                log_dir = os.path.join(root, f)
                if os.path.isfile(os.path.join(log_dir, "test_result.csv")):
                    result = processing_log_dir(result_dir, log_dir)
                    report_content += " ".join(result["options"]) + ","
                    report_content += str(result["test_file_result"]["test_accuracy"]) + ","
                    report_content += str(result["log_file_result"]["train_loss_at_max_iter"]) + ","
                    report_content += str(result["log_file_result"]["test_loss_at_max_iter"]) + ","
                    report_content += str(result["log_file_result"]["accuracy_at_max_iter"]) + ","
                    report_content += str(result["log_file_result"]["max_accuracy"]) + ","
                    report_content += str(result["log_file_result"]["over_fitting"]) + ","
                    report_content += str(result["log_file_result"]["total_time"]) + "\n"

        with open('multi_result_report.csv', 'w') as the_file:
            the_file.write(report_content)
    else:
        raise Exception("Please specific result dir")
