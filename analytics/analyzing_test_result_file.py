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


class AnalyzingTestResult:

    def __init__(self):
        pass

    def __read_test_file(self, result_file):
        result = []
        with open(result_file, "r") as ins:
            for idx, line in enumerate(ins):
                if idx == 0:
                    continue
                content = line.strip().split(",")
                result.append({'id': content[0], 'label': content[1]})
        return result

    def __predict_correct(self, test, classes):
        actual = test["id"].split("_")[0]
        return classes.index(actual) == int(test["label"])

    def analyzing(self, result_file, classes):
        correct_count = 0
        result_test = self.__read_test_file(result_file)
        for test in result_test:
            if self.__predict_correct(test, classes):
                correct_count += 1
        accuracy = float(correct_count) / float(len(result_test))
        return {
            "test_accuracy" : accuracy
        }