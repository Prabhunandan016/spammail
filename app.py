import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("mail_data.csv", encoding="latin-1")

# Detect spam label & message columns
label_col = [col for col in df.columns if "label" in col.lower() or "category" in col.lower() or "v1" in col.lower()][0]
message_col = [col for col in df.columns if "message" in col.lower() or "text" in col.lower() or "v2" in col.lower()][0]

# Rename columns for consistency
df = df[[label_col, message_col]]
df.columns = ["Label", "Message"]
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42)

# Create Model Pipeline
model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(X_train, y_train)

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message to check if it's Spam or Not.")

# User Input
user_input = st.text_area("Enter your message:")
if st.button("Check Spam"):
    if user_input:
        prediction = model.predict([user_input])[0]
        st.write("### Prediction: **SPAM**" if prediction == 1 else "### Prediction: **HAM**")
    else:
        st.warning("Please enter a message.")

# Model Performance Metrics
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
st.pyplot(fig)
