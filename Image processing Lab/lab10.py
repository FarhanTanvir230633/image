
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    uniform_filter, gaussian_filter,
    median_filter, laplace
)

# Read image
img = cv2.imread('input.jpg')

if img is None:
    raise FileNotFoundError("Check the location of input.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32) / 255.0

# 1. Mean smoothing
mean_img = uniform_filter(gray, size=3)

# 2. Gaussian smoothing
gaussian_img = gaussian_filter(gray, sigma=1)

# 3. Median smoothing
median_img = median_filter(gray, size=3)

# 4. Laplacian sharpening
lap_response = laplace(gray)
lap_img = np.clip(gray - lap_response, 0, 1)

# 5. High-boost sharpening
blur_img = gaussian_filter(gray, sigma=1)
A = 1.5
highboost_img = np.clip(A * gray - blur_img, 0, 1)

# Display results
images = [
    gray, mean_img, gaussian_img,
    median_img, lap_img, highboost_img
]

titles = [
    'Original Image',
    'Mean Smoothing',
    'Gaussian Smoothing',
    'Median Smoothing',
    'Laplacian Sharpening',
    'High-Boost Sharpening'
]

plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
