import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io

with open("fastapi_app/idx_to_class.json") as f:
    idx_to_class = json.load(f)

NUM_CLASSES = len(idx_to_class)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "fastapi_app/skin_disease_model_state_dict.pth"
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.model = base_model
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

def load_model():
    base_model = models.efficientnet_b0(pretrained=False)
    model = CustomModel(base_model)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229] * 3)
])

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = idx_to_class[str(pred.item())]
    return predicted_class
