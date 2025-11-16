import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE

def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, test_loader, dataset.classes
