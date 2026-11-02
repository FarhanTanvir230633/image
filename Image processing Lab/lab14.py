
import speech_recognition as sr

# Create recognizer object
recognizer = sr.Recognizer()

# Load audio file
audio_file = "speech.wav"

try:
    with sr.AudioFile(audio_file) as source:

        print("Reading audio file...")

        audio = recognizer.record(source)

    # Recognize speech
    text = recognizer.recognize_google(audio)

    print("Recognized Speech:")
    print(text)

except sr.UnknownValueError:
    print("Speech could not be understood.")

except sr.RequestError:
    print("Internet connection or service error.")

except FileNotFoundError:
    print("Audio file not found.")
