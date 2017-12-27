import os


class Workspace:
    curr_workspace = ""

    def __init__(self, dir_path):
        self.curr_workspace = dir_path
        pass

    def workspace(self, path):
        dir_path = self.curr_workspace + "/" + path
        if os.path.isdir(dir_path):
            return dir_path
        file_dir = os.path.dirname(dir_path)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        return dir_path