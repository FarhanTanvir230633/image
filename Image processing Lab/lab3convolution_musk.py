
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# 1. Read the image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# 2. Define a 3x3 averaging mask
mask = np.ones((3, 3), dtype=np.float32) / 9

# 3. Perform convolution
result = convolve2d(
    img.astype(np.float32),
    mask,
    mode="same",
    boundary="symm"
)

# 4. Convert the result for display
result = np.clip(result, 0, 255).astype(np.uint8)

# 5. Display original image and convolution result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("3x3 Convolution Result")
plt.axis("off")

plt.tight_layout()
plt.show()

print("3x3 convolution completed successfully.")
