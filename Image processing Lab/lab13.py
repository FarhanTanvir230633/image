
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Read speech signal
x, fs = sf.read("speech.wav")

# Convert stereo to mono
if x.ndim == 2:
    x = np.mean(x, axis=1)

# Select a short segment for clear plotting
N = min(len(x), int(0.05 * fs))
x = x[:N]

# Normalize amplitude
x = x / (np.max(np.abs(x)) + 1e-12)

# Quantization levels
levels = [4, 8, 16, 32]

# Time axis
t = np.arange(len(x)) / fs

# Plot original and quantized signals
plt.figure(figsize=(12, 9))

plt.subplot(3, 2, 1)
plt.plot(t, x)
plt.title("Original Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

for i, L in enumerate(levels):
    xmin, xmax = -1.0, 1.0
    step = (xmax - xmin) / (L - 1)

    # Uniform quantization
    xq = xmin + np.round((x - xmin) / step) * step
    xq = np.clip(xq, xmin, xmax)

    plt.subplot(3, 2, i + 2)
    plt.step(t, xq, where="mid")
    plt.title(f"Quantized Speech: {L} Levels ({int(np.log2(L))} bits)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.ylim(-1.1, 1.1)
    plt.grid(True)

    # Mean squared quantization error
    mse = np.mean((x - xq) ** 2)
    print(f"{L} levels ({int(np.log2(L))} bits): MSE = {mse:.6f}")

plt.tight_layout()
plt.show()
