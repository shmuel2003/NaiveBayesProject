from flask import Flask, request, jsonify

from data_cleaner import DataCleaner
from classifier import Classifier
from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer

app = Flask(__name__)

dataset_loaded = False

@app.route("/train", methods=["POST"])
def train_model():
    global dataset_loaded
    filename = request.json.get("filename")
    label_field = request.json.get("label_field")
    try:
        loader = DataLoader(filename)
        data = loader.load_data()
        DataCleaner.clean(data)
        train_data, test_data = loader.split_data(data)
        Trainer.train(train_data, label_field)
        app.test_data = test_data
        dataset_loaded = True
        return jsonify({"message": "Model trained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/evaluate", methods=["GET"])
def evaluate_model():
    if not dataset_loaded:
        return jsonify({"error": "Model not trained yet."}), 400

    accuracy = Evaluator.evaluate(app.test_data)
    return jsonify({"accuracy": accuracy})

@app.route("/predict", methods=["POST"])
def predict_instance():
    if not dataset_loaded:
        return jsonify({"error": "Model not trained yet."}), 400
    instance = request.json.get("instance")
    prediction = Classifier.classify(instance)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
