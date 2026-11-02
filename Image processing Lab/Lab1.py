
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# 2. Calculate the image histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 3. Apply a 3x3 low-pass averaging filter
low_kernel = np.ones((3, 3), dtype=np.float32) / 9
low_pass = cv2.filter2D(img, -1, low_kernel)

# 4. Apply a high-pass filter
high_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

high_raw = cv2.filter2D(
    img.astype(np.float32), -1, high_kernel
)

# Scale absolute values for display
high_pass = cv2.convertScaleAbs(high_raw)

# 5. Display all results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.plot(hist, color="black")
plt.title("Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.imshow(low_pass, cmap="gray", vmin=0, vmax=255)
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass, cmap="gray", vmin=0, vmax=255)
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()
