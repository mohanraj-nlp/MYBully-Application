import joblib
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Load Models and Encoders ---
emotion_tokenizer = AutoTokenizer.from_pretrained("mohanrj/mybully-emobert-manual")
emotion_model = AutoModelForSequenceClassification.from_pretrained("mohanrj/mybully-emobert-manual")

hate_tokenizer = AutoTokenizer.from_pretrained("mohanrj/mybully-hatebert-manual")
hate_model = AutoModelForSequenceClassification.from_pretrained("mohanrj/mybully-hatebert-manual")

sent_tokenizer = AutoTokenizer.from_pretrained("mohanrj/mybully-sentibert-manual")
sent_model = AutoModelForSequenceClassification.from_pretrained("mohanrj/mybully-sentibert-manual")

cyber_model = joblib.load("models/cyber_model.pkl")
hate_encoder = joblib.load("models/hate_encoder.pkl")
emo_encoder = joblib.load("models/emo_encoder.pkl")
sent_encoder = joblib.load("models/sent_encoder.pkl")

# --- Your training-time label mappings ---
emotion_label2id = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happniess': 3,
    'Neutral': 4,
    'Sadness': 5,
    'Surprise': 6
}

hatespeech_label2id = {
    'Yes': 1,
    'No': 0
}

sentiment_label2id = {
    'Negative': 0, 
    'Neutral': 1, 
    'Positive': 2
}

# --- Step 1: Invert label2id to get model label decoding ---
id2emotion_label = {f'LABEL_{v}': k for k, v in emotion_label2id.items()}
id2hate_label = {f'LABEL_{v}': k for k, v in hatespeech_label2id.items()}
id2sent_label = {f'LABEL_{v}': k for k, v in sentiment_label2id.items()}

# --- Predict Emotion from text ---
def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    label_id = emotion_model.config.id2label[pred_id]  
    return id2emotion_label.get(label_id, "Unknown")

# --- Predict HateSpeech from text ---
def predict_hate(text):
    inputs = hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = hate_model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    label_id = hate_model.config.id2label[pred_id] 
    return id2hate_label.get(label_id, "Unknown")

# --- Predict Sentiment from text ---
def predict_sentiment(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    label_id = sent_model.config.id2label[pred_id]
    return id2sent_label.get(label_id, "Unknown")

bert_tokenizer = AutoTokenizer.from_pretrained("mesolitica/roberta-base-bahasa-cased")
bert_model = AutoModel.from_pretrained("mesolitica/roberta-base-bahasa-cased")

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    return hidden_states.mean(dim=1).squeeze().numpy()  # (hidden_dim,)

# --- Extract combined features ---
def extract_features_single(tweet, hate_label, emo_label, sent_label):
    bert_feat = get_bert_embedding(tweet).reshape(1, -1)  # shape: (1, hidden_dim)
    hate_feat = hate_encoder.transform([[hate_label]])  # shape: (1, h)
    emo_feat = emo_encoder.transform([[emo_label]])    # shape: (1, e)
    sent_feat = sent_encoder.transform([[sent_label]])  # shape: (1, s)
    return np.hstack([bert_feat, hate_feat, emo_feat, sent_feat])  # shape: (1, total_dim)

# --- Predict Cyberbully ---
def predict_cyberbully(tweet, emotion, hate, sentiment):
    X = extract_features_single(tweet, hate, emotion, sentiment)
    pred = cyber_model.predict(X)[0]
    return "Cyberbully" if pred == 1 else "Not Cyberbully"

# --- Risk Meter (optional heuristic) ---
def get_risk_level(emotion, hate, sentiment):
    high_risk_emotions = ["Anger", "Disgust"]
    medium_risk_emotions = ["Sadness", "Fear"]

    if emotion in high_risk_emotions and hate == "Yes":
        return "High"
    elif emotion in medium_risk_emotions and hate == "Yes":
        return "Medium"
    else:
        return "Low"