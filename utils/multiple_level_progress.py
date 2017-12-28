import sys

from percent_visualize import print_progress


class MultipleLevelProgress:
    num_progress = 0
    max_val = 0

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
            MultipleLevelProgress.down()
            self.max_val = max_val
        pass

    def update(self, progress_index, val, message):
        for i in range(self.num_progress - progress_index):
            MultipleLevelProgress.up()
        prefix = "Processing (" + str(progress_index) + "/" + str(self.num_progress) + ") "
        print_progress(val, self.max_val, prefix, message, bar_length=50)
        for i in range(self.num_progress - progress_index):
            MultipleLevelProgress.down()

