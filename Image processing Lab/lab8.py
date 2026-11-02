
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# 1. Read the image
img = cv2.imread("coins.jpg")

if img is None:
    raise FileNotFoundError("coins.png not found!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Smooth the image and apply Otsu thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Select the polarity that makes the coins foreground
border = np.concatenate([
    binary[0, :], binary[-1, :],
    binary[:, 0], binary[:, -1]
])
if np.mean(border == 255) < 0.5:
    binary = cv2.bitwise_not(binary)

# 3. Remove small noise and fill holes
mask = binary > 0
mask = ndi.binary_fill_holes(mask)
mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
mask = ndi.binary_fill_holes(mask)

# Remove small foreground components
labels0, n0 = ndi.label(mask)
sizes = np.bincount(labels0.ravel())
mask = mask & (sizes[labels0] >= 100)

# 4. Separate touching coins using watershed
distance = ndi.distance_transform_edt(mask)
local_max = cv2.dilate(
    distance.astype(np.float32), np.ones((3, 3), np.uint8)
)
peaks = (distance == local_max) & mask
markers, _ = ndi.label(peaks)

labels = cv2.watershed(
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    np.zeros(gray.shape, dtype=np.int32)
) if False else None

# Watershed on the distance map
from skimage.segmentation import watershed
labels = watershed(-distance, markers, mask=mask)

# 5. Measure and label each coin
output = img.copy()
areas = []
coin_id = 0

for region_id in np.unique(labels):
    if region_id == 0:
        continue

    region = labels == region_id
    area = int(np.sum(region))

    if area < 100:
        continue

    coin_id += 1
    areas.append(area)

    ys, xs = np.where(region)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # Draw the coin boundary and label
    contours, _ = cv2.findContours(
        region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    cv2.putText(
        output, str(coin_id), (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    print(f"Coin {coin_id}: Area = {area} pixels^2")

print("Total coins detected:", coin_id)

# 6. Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Segmented Coins")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Labeled Coins and Areas")
plt.axis("off")

plt.tight_layout()
plt.show()

# 7. Display an area chart
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(areas) + 1), areas)
plt.xlabel("Coin Number")
plt.ylabel("Area (pixels^2)")
plt.title("Area of Each Coin")
plt.show()
