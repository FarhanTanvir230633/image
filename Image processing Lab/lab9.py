
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Read the input image
img = cv2.imread('input.jpg')

if img is None:
    raise FileNotFoundError("Please check the input.jpg file path.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# i) Threshold Image
T = 128
threshold_img = np.where(gray >= T, 255, 0).astype(np.uint8)

# ii) Power-Law Contrast Enhancement
gamma = 0.5
gray_float = gray.astype(np.float32) / 255.0
power_img = np.power(gray_float, gamma)
power_img = np.clip(power_img * 255, 0, 255).astype(np.uint8)

# iii) High-Pass Filtering
mask = np.array([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]], dtype=np.float32)

highpass_img = convolve(gray.astype(np.float32), mask)

# Normalize the high-pass output for display
hp_min = highpass_img.min()
hp_max = highpass_img.max()

if hp_max > hp_min:
    highpass_display = (
        (highpass_img - hp_min) / (hp_max - hp_min) * 255
    ).astype(np.uint8)
else:
    highpass_display = np.zeros_like(gray)

# Display results
images = [
    gray,
    threshold_img,
    power_img,
    highpass_display
]

titles = [
    'Original Grayscale Image',
    'Threshold Image',
    'Power-Law Enhanced Image',
    'High-Pass Image'
]

plt.figure(figsize=(10, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
