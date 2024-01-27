from flask import Flask, jsonify, render_template, request
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from io import BytesIO

app = Flask(__name__)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

checkpoint_path = "model/modelOCR.pth"
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:

        image_file = request.files['imageFile']

        image = Image.open(image_file)

        prediction_result = "Placeholder Prediction"

        return jsonify({'prediction': prediction_result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
