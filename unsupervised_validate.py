from utils.utils import *
import csv

set_workspace("data/unsupervised_mnist")

result = workspace("result/test_result.csv")
f = open(result, 'rb')
reader = csv.reader(f)
summary = {}
idx = 0
for row in reader:
    idx += 1
    if idx == 1:
        continue
    actual = str(row[2])
    predict = str(row[1])
    if actual in summary:
        if predict in summary[actual]:
            summary[actual][predict] += 1
        else:
            summary[actual][predict] = 1
    else:
        summary[actual] = {}
print summary