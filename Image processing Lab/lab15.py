
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Text-to-Speech Conversion
text = "Hello. This is a text to speech conversion experiment."

engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Convert text to speech and save it
engine.save_to_file(text, "tts_output.wav")
engine.runAndWait()

print("TTS audio saved as tts_output.wav")

# Part 2: Record Speech Using Microphone
fs = 16000
duration = 5

print("Speak into the microphone...")
recording = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1,
    dtype="float32"
)
sd.wait()

# Save recorded speech
sf.write("recorded_speech.wav", recording, fs)
print("Recording saved as recorded_speech.wav")

# Part 3: Plot the Recorded Speech
speech = recording.flatten()
time = np.arange(len(speech)) / fs

plt.figure(figsize=(10, 4))
plt.plot(time, speech)
plt.title("Recorded Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
