import os
import sys
from analyzing_log_file import AnalyzingLogFile

analyzing_log_file = AnalyzingLogFile()

def processing_log_dir(result_dir, log_dir):
    relation_path = log_dir.replace(result_dir + "/", '')
    options = relation_path.split("/")
    log_file = os.path.join(log_dir, "train.log")
    result_file = os.path.join(log_dir, "result.csv")

    log_file_result = analyzing_log_file.analyzing(log_file)

if __name__ == '__main__':
    result_dir = sys.argv[1]
    if os.path.isdir(result_dir):
        result_dir = os.path.abspath(result_dir)
        for root, dirs, files in os.walk(result_dir):
            for f in dirs:
                log_dir = os.path.join(root, f)
                if os.path.isfile(os.path.join(log_dir, "result.csv")):
                    processing_log_dir(result_dir, log_dir)

    else:
        raise Exception("Please specific result dir")
