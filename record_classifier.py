class RecordClassifier:
    def __init__(self, classifier, target_index):
        self.classifier = classifier
        self.target_index = target_index

    def classify(self, record):
        return self.classifier.predict(record)