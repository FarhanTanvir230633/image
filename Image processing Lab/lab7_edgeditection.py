
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the image in grayscale
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

img = img.astype(np.float32)

# 2. Roberts masks
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

rx = cv2.filter2D(img, cv2.CV_32F, roberts_x)
ry = cv2.filter2D(img, cv2.CV_32F, roberts_y)
roberts = np.sqrt(rx**2 + ry**2)

# 3. Prewitt masks
prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

px = cv2.filter2D(img, cv2.CV_32F, prewitt_x)
py = cv2.filter2D(img, cv2.CV_32F, prewitt_y)
prewitt = np.sqrt(px**2 + py**2)

# 4. Sobel operator
sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
sobel = np.sqrt(sx**2 + sy**2)

# 5. Laplacian operator
laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
laplacian = np.abs(laplacian)

# 6. Normalize results for display
def normalize(image):
    return cv2.normalize(
        image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

results = [
    ("Original Image", img),
    ("Roberts Operator", normalize(roberts)),
    ("Prewitt Operator", normalize(prewitt)),
    ("Sobel Operator", normalize(sobel)),
    ("Laplacian Operator", normalize(laplacian))
]

# 7. Display and compare results
plt.figure(figsize=(12, 8))

for i, (title, image) in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Edge detection using four operators completed.")
