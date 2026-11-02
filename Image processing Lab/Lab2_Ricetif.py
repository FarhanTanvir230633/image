
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, segmentation
from scipy import ndimage as ndi

# Experiment No. 02: Rice Grain Analysis

# 1. Read the image
img = io.imread(r"D:\Academic File\3-2\Digital Image and Speech processing\2026\Image processing Lab\rice (1).tif", as_gray=True)
# Convert to an 8-bit grayscale image
img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

# 2. Threshold the image
T = filters.threshold_otsu(img)
binary = img > T

# Ensure rice grains are the foreground objects
if binary.mean() > 0.5:
    binary = ~binary

# 3. Remove small noise and fill holes
binary = morphology.remove_small_objects(binary, min_size=20)
binary = ndi.binary_fill_holes(binary)

# 4. Separate touching grains using watershed
distance = ndi.distance_transform_edt(binary)
local_max = morphology.local_maxima(distance)
markers, _ = ndi.label(local_max)
labels = segmentation.watershed(
    -distance, markers, mask=binary
)

# 5. Measure each rice grain
props = measure.regionprops(labels)

# Remove very small regions if necessary
props = [p for p in props if p.area >= 20]

print("Total number of rice grains:", len(props))
print("\nGrain  Area (pixels^2)  Major Axis (pixels)  Perimeter (pixels)")

for i, p in enumerate(props, start=1):
    print(f"{i:5d} {p.area:15.2f} {p.axis_major_length:20.2f} {p.perimeter:19.2f}")

# 6. Select grains within a specific area range
min_area = 100
max_area = 500

selected = [
    p for p in props
    if min_area <= p.area <= max_area
]

print(f"\nGrains with area from {min_area} to {max_area} pixels^2:")
print("Number of selected grains:", len(selected))

for i, p in enumerate(selected, start=1):
    print(
        f"Selected {i}: Area={p.area:.2f}, "
        f"Major Axis={p.axis_major_length:.2f}, "
        f"Perimeter={p.perimeter:.2f}"
    )

# 7. Display the results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original Rice Image")
ax[0].axis("off")

ax[1].imshow(binary, cmap="gray")
ax[1].set_title("Binary Segmented Image")
ax[1].axis("off")

ax[2].imshow(img, cmap="gray")
ax[2].set_title("Rice Grain Labels")
ax[2].axis("off")

for p in props:
    y, x = p.centroid
    ax[2].text(x, y, str(p.label), color="red",
               fontsize=7, ha="center")

plt.tight_layout()
plt.show()
