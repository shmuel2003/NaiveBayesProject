from typing import Dict

class PredictionMixin:
    def predict(self, item: Dict[str, str]) -> str:
        label_scores = {}
        for label in self.classes:
            prob = self.class_probs[label]
            for feature, value in item.items():
                prob *= self.feature_probs[feature][value].get(label, 1e-6)
            label_scores[label] = prob
        return max(label_scores, key=label_scores.get)
