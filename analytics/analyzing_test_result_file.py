
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