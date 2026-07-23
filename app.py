import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vect = joblib.load("vect.pkl")   # Agar vectorizer.pkl save kiya hai to uska naam use karo

# Title
st.title("📝 Sentiment Analysis App")

st.write("Enter any sentence below.")

# Text Input
text = st.text_area("Enter Text")

# Predict Button
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert text into TF-IDF vector
        text_vector = vect.transform([text])

        # Predict sentiment
        prediction = model.predict(text_vector)

        # Display result
        if prediction[0] == "Positive":
            st.success("🟢 Positive")

        elif prediction[0] == "Negative":
            st.error("🔴 Negative")

        elif prediction[0] == "Neutral":
            st.info("🟡 Neutral")