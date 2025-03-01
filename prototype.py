from textblob import TextBlob
from transformers import pipeline,AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load AI-based sentiment and emotion analysis models
sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load Grammarly's grammar correction model
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large")

def correct_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_beams=5)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text



def analyze_sentence(sentence):
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    sentiment_result = sentiment_pipeline(sentence)[0]['label']
    emotion_result = emotion_pipeline(sentence)[0]['label']

    factuality = "Factual" if subjectivity < 0.5 else "Personal"
    negation = "Negation" if "not" in sentence.lower() or "n't" in sentence.lower() else "Affirmation"

    return {
        "Corrected Sentence": correct_sentence(sentence),
        "Polarity": polarity,
        "Subjectivity": subjectivity,
        "Sentiment": sentiment_result,
        "Emotion": emotion_result,
        "Factuality": factuality,
        "Type": negation
    }


if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    result = analyze_sentence(sentence)
    for key, value in result.items():
        print(f"{key}: {value}")