from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = Flask(__name__)


model = YOLO("/home/yousef/Fashion Recognation/runs/classify/train/weights/best.pt") 

# Mapping of Fashion-MNIST class indices to labels
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale if needed
        img = img.resize((32, 32))  # Ensure size matches model input size

        # Convert grayscale to RGB if model expects 3 channels
        img_rgb = Image.merge("RGB", (img, img, img))

        # Normalize image to [0, 1] range if required
        img_array = (np.array(img_rgb) * 255).astype(np.uint8)

        # Run inference
        results = model(img_array)

        # Extract predictions from Results object
        predictions = []
        for result in results:
            top_class_index = int(result.probs.top1)  # Get top class index
            confidence_score = float(result.probs.top1conf)  # Get confidence score
            
            # Map class index to class name
            class_name = class_labels.get(top_class_index, "Unknown")

            predictions.append({
                "class": class_name,
                "confidence": confidence_score
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
