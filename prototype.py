from textblob import TextBlob
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load AI-based sentiment and emotion analysis models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load AI-based factuality model using BART-based NLI
factuality_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load AI-based negation detection model
negation_pipeline = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")

# Load BART-based grammar correction model
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large")


def correct_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_beams=5)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text


def classify_factuality(sentence):
    labels = ["fact", "opinion", "false"]  # Possíveis categorias
    result = factuality_pipeline(sentence, candidate_labels=labels)
    return result["labels"][0]  # Retorna a categoria mais provável


def analyze_sentence(sentence):
    # Análise de polaridade e subjetividade (TextBlob)
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Análise de sentimento (positivo/negativo) e emoção
    sentiment_result = sentiment_pipeline(sentence)[0]['label']
    emotion_result = emotion_pipeline(sentence)[0]['label']

    # Classificação de factualidade
    factuality_result = classify_factuality(sentence)

    # Detecção simples de negação (usando modelo de sentimento como proxy)
    negation_result = negation_pipeline(sentence)
    negation = "Negation" if negation_result[0]['label'] == 'negative' else "Affirmation"

    # Correção gramatical
    corrected_sentence = correct_sentence(sentence)

    return {
        "Original Sentence": sentence,
        "Corrected Sentence": corrected_sentence,
        "Polarity (TextBlob)": polarity,
        "Subjectivity (TextBlob)": subjectivity,
        "Sentiment (HF Pipeline)": sentiment_result,
        "Emotion (HF Pipeline)": emotion_result,
        "Factuality": factuality_result,
        "Type": negation
    }


if __name__ == "__main__":
    # Lista de frases para teste (algumas bem escritas, outras com erros)
    sentences = [
        "I love going to the park on sunny days.",
        "I realy enyoj to read boks about psychology.",
        "He will tri to parse that code untill it works.",
        "The Earth is round.",
        "I do not think it's correct.",
        "He never said that he would come."
    ]

    for sentence in sentences:
        print("------------------------------------------------")
        result = analyze_sentence(sentence)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("------------------------------------------------\n")
