
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the text image in grayscale
img = cv2.imread(r"D:\Academic File\3-2\Digital Image and Speech processing\2026\Image processing Lab\hello.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# 2. Convert the image into binary format
# Dark characters become white foreground pixels
_, binary = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 3. Find connected components
count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(binary, 8)
)

characters = []

for i in range(1, count):
    x, y, w, h, area = stats[i]

    # Remove tiny noise components
    if area >= 10 and h >= 10 and w >= 2:
        characters.append((x, y, w, h))

# 4. Sort characters from left to right
characters.sort(key=lambda item: item[0])

# 5. Display original and binary images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Text Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# 6. Extract and display individual characters
print("Number of detected character components:", len(characters))

fig, axes = plt.subplots(
    1, max(1, len(characters)),
    figsize=(2 * max(1, len(characters)), 3),
    squeeze=False
)

for i, (x, y, w, h) in enumerate(characters):
    char_img = binary[y:y+h, x:x+w]

    axes[0, i].imshow(char_img, cmap="gray")
    axes[0, i].set_title(f"Component {i+1}")
    axes[0, i].axis("off")

    # Save each extracted component
    cv2.imwrite(f"character_{i+1}.png", char_img)

plt.tight_layout()
plt.show()

# 7. Display character boundaries on the original image
output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for i, (x, y, w, h) in enumerate(characters):
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 1)

plt.figure(figsize=(10, 4))
plt.imshow(output)
plt.title("Segmented Character Regions")
plt.axis("off")
plt.show()
