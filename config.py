import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4

NUM_CLASSES = 4  # background + person + car + dog

TRAIN_IMAGES = "data/train/images"
TRAIN_MASKS = "data/train/masks"

VAL_IMAGES = "data/val/images"
VAL_MASKS = "data/val/masks"

TEST_IMAGES = "data/test/images" 
TEST_MASKS  = "data/test/masks"   

MODEL_PATH = "unet_model.pth"