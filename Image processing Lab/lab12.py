import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import medfilt

fs, x = wavfile.read("speech.wav")
x = x.astype(float)
if x.ndim > 1:
    x = x.mean(axis=1)
x /= np.max(np.abs(x))

N, H = int(0.025 * fs), int(0.010 * fs)
n = 1 + (len(x) - N) // H
frames = np.array([x[i * H:i * H + N] for i in range(n)])

energy = np.sum(frames ** 2, axis=1)
energy /= energy.max()
zcr = 0.5 * np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1)

sil_th = 0.01
zcr_th = np.median(zcr[energy > sil_th])

label = np.where(energy < sil_th, 0, np.where(zcr < zcr_th, 2, 1))
label = medfilt(label, 5).astype(int)

colors = {0: "blue", 1: "red", 2: "green"}
names = {0: "Silence", 1: "Unvoiced", 2: "Voiced"}
t = np.arange(len(x)) / fs

plt.figure(figsize=(12, 4))
for i, l in enumerate(label):
    s = i * H
    plt.plot(t[s:s + H + 1], x[s:s + H + 1], color=colors[l])
for l in colors:
    plt.plot([], [], color=colors[l], label=names[l])
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Voiced / Unvoiced / Silence")
plt.show()