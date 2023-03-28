import torch.cuda

NUM_EPOCHS = 1
BATCH_SIZE = 32
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0
LEARNING_RATE_GEN = 1e-5
LEARNING_RATE_DISC = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
