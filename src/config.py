import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "animals")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

NUM_CLASSES = 15
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0003
CLASSES = [
    "Bear",
    "Bird",
    "Cat",
    "Cow",
    "Deer",
    "Dog",
    "Dolphin",
    "Elephant",
    "Giraffe",
    "Horse",
    "Kangaroo",
    "Lion",
    "Panda",
    "Tiger",
    "Zebr"
]
