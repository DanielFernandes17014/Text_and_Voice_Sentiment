import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from transformers import pipeline
from textblob import TextBlob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize speech recognizer
recognizer = sr.Recognizer()


# Function to plot and update the spectrogram
def update_spectrogram():
    fs = 44100  # Sampling frequency
    duration = 3  # Duration of recording
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    plt.clf()
    plt.specgram(recording[:, 0], Fs=fs, cmap='inferno')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    canvas.draw()
    return recording


def transcribe_speech():
    with sr.Microphone() as source:
        btn_listen.config(text="Listening...", state=tk.DISABLED)
        root.update()
        recording = update_spectrogram()
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    btn_listen.config(text="Start Listening", state=tk.NORMAL)
    root.update()

    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed: {text}")
        analyze_text(text)
    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand audio")
    except sr.RequestError:
        messagebox.showerror("Error", "Speech recognition service is unavailable")


# Load AI-based sentiment and emotion analysis models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load AI-based factuality model using BART-based NLI
factuality_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def analyze_text(text):
    # Sentiment & Emotion
    sentiment = sentiment_pipeline(text)[0]
    emotion = emotion_pipeline(text)[0]

    # Factuality analysis
    factuality = factuality_pipeline(text, candidate_labels=["FACT", "NEUTRAL", "CONTRADICTION"])['labels'][0]

    # Polarity & Subjectivity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Display results in a message box and console
    output = (f"Polarity: {polarity}\nSubjectivity: {subjectivity}\n"
              f"Sentiment: {sentiment['label']}\nEmotion: {emotion['label']}\nFactuality: {factuality}")
    print(output)


# GUI setup
root = tk.Tk()
root.title("Speech Analyzer")

fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

btn_listen = tk.Button(root, text="Start Listening", command=transcribe_speech, padx=20, pady=10)
btn_listen.pack(pady=20)

root.mainloop()
