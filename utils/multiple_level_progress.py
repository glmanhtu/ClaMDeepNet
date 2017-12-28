import progressbar
import sys


class MultipleLevelProgress:
    num_progress = 0
    progressbars = []

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
            current_progress = progressbar.ProgressBar(maxval=max_val)
            current_progress.start()
            self.progressbars.append(current_progress)
        pass

    def update(self, progress_index, val):
        for i in range(self.num_progress - progress_index):
            MultipleLevelProgress.up()
        self.progressbars[progress_index].update(val)
        for i in range(self.num_progress - progress_index):
            MultipleLevelProgress.down()

