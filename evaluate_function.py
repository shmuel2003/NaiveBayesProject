from typing import List, Dict

class EvaluationMixin:
    def evaluate(self, test_data: List[Dict[str, str]]) -> float:
        correct = 0
        for row in test_data:
            true_label = row[self.label_field]
            features = {k: v for k, v in row.items() if k != self.label_field}
            predicted = self.predict(features)
            if predicted == true_label:
                correct += 1
        return correct / len(test_data) if test_data else 0.0
