import sys
import torch
from PIL import Image
from torchvision import transforms
from model import get_model   # FIXED

# Load model
model = get_model()
model.load_state_dict(torch.load("animal_model.pth", map_location="cpu", weights_only=True))
model.eval()

# Class names
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Dog",
    "Elephant", "Giraffe", "Horse", "Lion",
    "Panda", "Sheep", "Tiger", "Wolf", "Zebra", "Monkey"
]

def predict(img_path):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)

    print("Predicted class:", class_names[predicted.item()])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
