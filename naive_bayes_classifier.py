from collections import defaultdict
from typing import List, Dict
from predict_function import PredictionMixin
from evaluate_function import EvaluationMixin

class NaiveBayesClassifier(PredictionMixin, EvaluationMixin):
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes = set()
        self.label_field = None

    def train(self, data: List[Dict[str, str]], label_field: str):
        self.label_field = label_field
        label_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        total = len(data)

        for row in data:
            label = row[label_field]
            label_counts[label] += 1
            self.classes.add(label)
            for feature, value in row.items():
                if feature == label_field:
                    continue
                feature_counts[feature][value][label] += 1

        for label in self.classes:
            self.class_probs[label] = (label_counts[label] + 1) / (total + len(self.classes))

        for feature, value_dict in feature_counts.items():
            for value, label_dict in value_dict.items():
                for label in self.classes:
                    count = label_dict.get(label, 0)
                    self.feature_probs[feature][value][label] = (count + 1) / (label_counts[label] + len(value_dict))
