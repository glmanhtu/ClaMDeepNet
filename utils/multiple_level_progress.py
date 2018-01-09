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

import sys

from percent_visualize import print_progress


class MultipleLevelProgress:
    num_progress = 0
    max_val = 0
    max_pos = 0
    progresses = []

    @staticmethod
    def up():
        # My terminal breaks if we don't flush after the escape-code
        sys.stdout.write('\x1b[1A')
        sys.stdout.flush()

    @staticmethod
    def down():
        # I could use '\x1b[1B' here, but newline is faster and easier
        sys.stdout.write('\n')
        sys.stdout.flush()

    def __init__(self, num_progress, max_val):
        self.num_progress = num_progress
        for progress in range(num_progress):
            self.max_val = max_val
        MultipleLevelProgress.down()
        pass

    def get_progress_index(self, test_id):
        if test_id not in self.progresses:
            self.progresses.append(test_id)
        return self.progresses.index(test_id) + 1

    def update(self, test_id, val, message):
        progress_index = self.get_progress_index(test_id)

        if progress_index > self.max_pos:
            for i in range(progress_index - self.max_pos):
                MultipleLevelProgress.down()
        else:
            for i in range(self.max_pos - progress_index):
                MultipleLevelProgress.up()

        if progress_index >= self.max_pos:
            self.max_pos = progress_index + 1

        prefix = "ID: " + str(test_id) + " |--> (" + str(progress_index) + "/" + str(self.num_progress) + ") "
        print_progress(val, self.max_val, prefix, message, bar_length=30)

        for i in range(self.max_pos - progress_index):
            MultipleLevelProgress.down()

