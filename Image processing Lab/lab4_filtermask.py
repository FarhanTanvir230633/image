
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# Convert to floating-point for filtering
img_float = img.astype(np.float32)

# 2. Define the Laplacian filter mask
mask = np.array([
    [0, -1,  0],
    [-1, 4, -1],
    [0, -1,  0]
], dtype=np.float32)

# 3. Apply the Laplacian filter
laplacian = cv2.filter2D(
    img_float, cv2.CV_32F, mask,
    borderType=cv2.BORDER_REFLECT
)

# 4. Create the sharpened image
sharpened = img_float + laplacian
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Scale Laplacian response for visualization
lap_display = cv2.normalize(
    np.abs(laplacian), None, 0, 255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# 5. Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(lap_display, cmap="gray")
plt.title("Laplacian Response")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap="gray")
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Laplacian filtering completed successfully.")
