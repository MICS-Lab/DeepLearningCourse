import torch.cuda

# Force flush
import functools
print = functools.partial(print, flush=True)  # noqa

# Global variables
FOLDER_FOR_VAL_PLOTS = "label_smoothing/images_with_skip_with_double_res_block_act_no_act"

NUM_EPOCHS = 301
BATCH_SIZE = 12
LAMBDA_CYCLE = 5
LAMBDA_IDENTITY = 0
LEARNING_RATE_GEN = 1e-5
LEARNING_RATE_DISC = 1e-5

LABEL_SMOOTHING = 0.1
"""
Label smoothing can help improve the convergence of CycleGAN by providing a more nuanced target distribution for the 
discriminator. In traditional GANs, the discriminator is trained to distinguish between real and fake samples by 
assigning a label of 1 to real samples and 0 to fake samples. This creates a hard boundary between real and fake 
samples, which can make the discriminator too confident in its predictions.
In contrast, label smoothing replaces the hard targets (e.g., 0 and 1) with smoothed targets that are slightly shifted 
towards each other. For example, instead of labeling real samples as 1 and fake samples as 0, we can label real samples 
as 0.9 and fake samples as 0.1. This makes the target distribution for the discriminator more nuanced and encourages 
the discriminator to be less certain about its predictions.
By using smoothed targets, we can prevent the discriminator from becoming too confident in its ability to distinguish 
real from fake samples. This, in turn, makes it more difficult for the generator to overfit to the discriminator and 
helps improve the stability and convergence of the GAN.
"""
USE_SKIP_CONNECTIONS = True
"""
We add the initial image to the output of the generator, before the tanh.
"""
DOUBLE_RES_BLOCKS = True
"""
Cf. generator_model.py
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
