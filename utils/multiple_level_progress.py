import sys

from percent_visualize import print_progress


class MultipleLevelProgress:
    num_progress = 0
    max_val = 0
    current_pos = 0

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

    def update(self, progress_index, val, message):
        change_pos = progress_index - self.current_pos

        if change_pos > 0:
            for i in range(abs(change_pos)):
                MultipleLevelProgress.down()
        else:
            for i in range(abs(change_pos)):
                MultipleLevelProgress.up()
        prefix = "Processing (" + str(progress_index) + "/" + str(self.num_progress) + ") "
        print_progress(val, self.max_val, prefix, message, bar_length=50)
        self.current_pos = progress_index

