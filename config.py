# User-tweakable parameters

USE_SIREN = True  # Toggle to use SIREN-based network

width = 400
height = 400
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5
NUM_FREQUENCIES = 10
# Each frequency gives 4 dimensions (sin/cos for both x and y)
NUM_ENCODED_DIMS = 4 * NUM_FREQUENCIES  # This will be 40 for NUM_FREQUENCIES = 10
NUM_HIDDEN = 128
NUM_OUTPUTS = 1
CHECKPOINT_DIR = "checkpoints"
VIDEO_DIR = "videos"

LOG_LEVEL = "INFO"

SHOW_NETWORK_OUTPUT = True       # If False, hide the network-generated surface (blue)
TARGET_ALPHA = 0.45               # Transparency for the target fractal (red), 0 fully transparent, 1 opaque

# Validate parameters
if width <= 0 or height <= 0:
    raise ValueError("Width and height must be positive.")
