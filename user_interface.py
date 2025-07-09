from data_loader import DataLoader
from naive_bayes_classifier import NaiveBayesClassifier
from record_classifier import RecordClassifier
from model_evaluator import ModelEvaluator

class UserInterface:
    def __init__(self):
        self.data_loader = DataLoader()
        self.classifier = NaiveBayesClassifier()
        self.target_index = None

    def run(self):
        print("=== Naive Bayes Classifier ===")
        train_path = input("Enter path to training CSV file: ")
        data = self.data_loader.load_csv(train_path)
        self.target_index = len(data[0]) - 1
        self.classifier.train(data, self.target_index)
        print("Model trained successfully.\n")

        while True:
            print("Choose an option:")
            print("1. Evaluate model on test CSV")
            print("2. Classify a single record")
            print("3. Exit")
            choice = input("Your choice: ")

            if choice == "1":
                test_path = input("Enter test CSV path: ")
                test_data = self.data_loader.load_csv(test_path)
                evaluator = ModelEvaluator(self.classifier, self.target_index)
                acc = evaluator.evaluate(test_data)
                print(f"Accuracy: {acc * 100:.2f}%\n")
            elif choice == "2":
                raw = input("Enter comma-separated feature values: ")
                record = raw.strip().split(",")
                record_classifier = RecordClassifier(self.classifier, self.target_index)
                prediction = record_classifier.classify(record)
                print(f"Predicted class: {prediction}\n")
            elif choice == "3":
                print("Goodbye.")
                break
            else:
                print("Invalid option.\n")