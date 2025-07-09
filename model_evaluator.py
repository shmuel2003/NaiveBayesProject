class ModelEvaluator:
    def __init__(self, classifier, target_index):
        self.classifier = classifier
        self.target_index = target_index

    def evaluate(self, test_data):
        correct = 0
        for row in test_data:
            actual = row[self.target_index]
            features = row[:self.target_index] + row[self.target_index+1:]
            predicted = self.classifier.predict(features)
            if predicted == actual:
                correct += 1
        accuracy = correct / len(test_data)
        return accuracy