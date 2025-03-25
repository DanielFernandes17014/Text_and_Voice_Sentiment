import speech_recognition as sr
import tkinter as tk
from transformers import pipeline
from textblob import TextBlob

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load AI-based sentiment and emotion analysis models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")
factuality_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Store transcriptions for batch processing
transcriptions = []


def transcribe_speech():
    with sr.Microphone() as source:
        btn_listen.config(text="Listening...", state=tk.DISABLED)
        root.update()
        recognizer.adjust_for_ambient_noise(source)
        print("Adjusting for ambient noise. Please wait.")
        audio = recognizer.listen(source, phrase_time_limit=10)

    btn_listen.config(text="Start Listening", state=tk.NORMAL)
    root.update()

    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed: {text}")
        transcriptions.append(text)
        if len(transcriptions) >= 5:  # Process in batches of 5
            analyze_text(transcriptions)
            transcriptions.clear()
    except sr.UnknownValueError:
        print("Error: Could not understand audio")
    except sr.RequestError:
        print("Error: Speech recognition service is unavailable")


def analyze_text(texts):
    # Sentiment & Emotion in batch
    sentiments = sentiment_pipeline(texts)
    emotions = emotion_pipeline(texts)

    # Factuality analysis (one by one since it's zero-shot)
    factualities = [factuality_pipeline(text, candidate_labels=["FACT", "NEUTRAL", "CONTRADICTION"])['labels'][0] for
                    text in texts]

    for i, text in enumerate(texts):
        # Polarity & Subjectivity
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Display results in console
        output = (f"Text: {text}\nPolarity: {polarity}\nSubjectivity: {subjectivity}\n"
                  f"Sentiment: {sentiments[i]['label']}\nEmotion: {emotions[i]['label']}\nFactuality: {factualities[i]}\n"
                  f"{'-' * 50}")
        print(output)


# GUI setup
root = tk.Tk()
root.title("Speech Analyzer")

btn_listen = tk.Button(root, text="Start Listening", command=transcribe_speech, padx=20, pady=10)
btn_listen.pack(pady=20)


# Cleanup on exit
def on_closing():
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
