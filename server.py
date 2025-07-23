from flask import Flask, request, jsonify
from data_loader import DataLoader
from naive_bayes_classifier import NaiveBayesClassifier

app = Flask(__name__)
classifier = NaiveBayesClassifier()
dataset_loaded = False

@app.route("/train", methods=["POST"])
def train_model():
    global dataset_loaded
    filename = request.json.get("filename")
    label_field = request.json.get("label_field")
    try:
        loader = DataLoader(filename)
        data = loader.load_data()
        train_data, test_data = loader.split_data(data)
        classifier.train(train_data, label_field)
        app.test_data = test_data
        dataset_loaded = True
        return jsonify({"message": "Model trained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/evaluate", methods=["GET"])
def evaluate_model():
    if not dataset_loaded:
        return jsonify({"error": "Model not trained yet."}), 400
    accuracy = classifier.evaluate(app.test_data)
    return jsonify({"accuracy": accuracy})

@app.route("/predict", methods=["POST"])
def predict_instance():
    if not dataset_loaded:
        return jsonify({"error": "Model not trained yet."}), 400
    instance = request.json.get("instance")
    prediction = classifier.predict(instance)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
