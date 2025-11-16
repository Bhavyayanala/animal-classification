import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import get_model

def train():
    train_loader, test_loader, classes = get_loaders()
    model = get_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Device:", device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(10):
        model.train()
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}: Train Accuracy = {100 * correct/total:.2f}%")

    torch.save(model.state_dict(), "animal_model.pth")
    print("Model saved as animal_model.pth")

if __name__ == "__main__":
    train()
