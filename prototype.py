from textblob import TextBlob
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def load_pipelines():
    return {
        "sentiment": [
            ("DistilBERT",
             pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")),
            ("RoBERTa", pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment"))
        ],
        "emotion": [
            ("GoEmotions", pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student"))
        ],
        "factuality": [
            ("BART", pipeline("text-classification", model="facebook/bart-large-mnli"))
        ],
        "negation": [
            ("RoBERTa Sentiment", pipeline("text-classification", model="siebert/sentiment-roberta-large-english"))
        ],
        "grammar_correction": {
            "tokenizer": AutoTokenizer.from_pretrained("grammarly/coedit-large"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large")
        }
    }


def correct_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def analyze_sentence(sentence, pipelines):
    corrected_sentence = correct_sentence(sentence, pipelines["grammar_correction"]["tokenizer"],
                                          pipelines["grammar_correction"]["model"])
    blob = TextBlob(corrected_sentence)
    polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity

    results = {
        "Original Sentence": sentence,
        "Corrected Sentence": corrected_sentence,
        "Polarity (TextBlob)": polarity,
        "Subjectivity (TextBlob)": subjectivity
    }

    for task, models in pipelines.items():
        if task in ["sentiment", "emotion", "factuality", "negation"]:
            for name, model in models:
                output = model(corrected_sentence)[0]
                results[f"{task.capitalize()} ({name})"] = f"{output['label']} ({output['score'] * 100:.2f}%)"

    return results


if __name__ == "__main__":
    sentences = [
        "the earth is round.",
        "i love her",
        "He will tyr to parse that code untill it works.",
        "The Earth is flat.",
        "I do not think it's correct.",
        "He never said that he would come.",
        "hy how hve u bene ? im so wrroied abut u we need to talk",
        "can u spto being rediculous ?",
        "i luv that about him, can yu belive that?"
    ]

    pipelines = load_pipelines()

    for sentence in sentences:
        print("------------------------------------------------")
        result = analyze_sentence(sentence, pipelines)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("------------------------------------------------\n")
