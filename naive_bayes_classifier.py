import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes = set()
        self.laplace = 1

    def train(self, dataset, target_index):
        total_count = len(dataset)
        class_counts = defaultdict(int)

        for row in dataset:
            cls = row[target_index]
            self.classes.add(cls)
            class_counts[cls] += 1

        for cls in self.classes:
            self.class_probs[cls] = (class_counts[cls] + self.laplace) / (total_count + self.laplace * len(self.classes))

        for row in dataset:
            cls = row[target_index]
            for i, value in enumerate(row):
                if i == target_index:
                    continue
                self.feature_probs[i][value][cls] += 1

        for feature in self.feature_probs:
            for value in self.feature_probs[feature]:
                for cls in self.classes:
                    count = self.feature_probs[feature][value][cls]
                    self.feature_probs[feature][value][cls] = \
                        (count + self.laplace) / (class_counts[cls] + self.laplace * len(self.feature_probs[feature]))

    def predict(self, record):
        scores = {}
        for cls in self.classes:
            score = math.log(self.class_probs[cls])
            for i, value in enumerate(record):
                prob = self.feature_probs[i].get(value, {}).get(cls, self.laplace / (1 + self.laplace))
                score += math.log(prob)
            scores[cls] = score
        return max(scores, key=scores.get)