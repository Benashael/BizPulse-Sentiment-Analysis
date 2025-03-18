import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    if text:
        textblob_score = TextBlob(text).sentiment.polarity
        vader_score = sia.polarity_scores(text)["compound"]
        
        if textblob_score > 0:
            sentiment = "Positive"
        elif textblob_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return textblob_score, vader_score, sentiment
    return 0, 0, "Neutral"

# Function for tokenization
def tokenize_text(text):
    return nltk.sent_tokenize(text)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="wide")
st.title("📊 Sentiment Analysis App")

# App Features
st.write("### Features of this Application:")
st.write("- **Single Text Sentiment Analysis**: Enter text manually to get sentiment results.")
st.write("- **Batch Sentiment Analysis**: Upload a CSV or TXT file with text data for bulk processing.")
st.write("- **Sentiment Visualization**: View sentiment score distributions through charts.")
st.write("- **Tokenization Support**: Upload a TXT file or enter text for tokenization and visualization.")

# Explanation of Scores
st.write("### Explanation of Sentiment Scores")
st.write("- **TextBlob Score**: Measures polarity of the text, ranging from -1 (negative) to +1 (positive).")
st.write("- **VADER Score**: Compound score that ranges from -1 (negative) to +1 (positive), suitable for short texts and social media.")

# User selects input type
input_type = st.radio("Select Input Type:", ("Text Input", "CSV File", "TXT File"))

if input_type == "Text Input":
    st.subheader("Enter Text for Sentiment Analysis")
    user_text = st.text_area("Type or paste text here:")
    if st.button("Analyze Sentiment"):
        textblob_score, vader_score, sentiment = analyze_sentiment(user_text)
        st.write(f"**TextBlob Score:** {textblob_score:.2f}")
        st.write(f"**VADER Score:** {vader_score:.2f}")
        st.write(f"**Predicted Sentiment:** {sentiment}")
        
        # Sentiment Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=["TextBlob", "VADER"], y=[textblob_score, vader_score], ax=ax, palette=["blue", "green"])
        ax.set_ylim([-1, 1])
        ax.set_ylabel("Sentiment Score")
        st.pyplot(fig)

elif input_type == "CSV File":
    st.subheader("Upload CSV File for Batch Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        column = st.selectbox("Select the text column", df.columns)
        if df[column].dtype == 'object':
            df["TextBlob_Score"], df["VADER_Score"], df["Sentiment"] = zip(*df[column].apply(analyze_sentiment))
            st.dataframe(df)
            
            # Sentiment Distribution Visualization
            fig, ax = plt.subplots()
            sns.countplot(x=df["Sentiment"], palette="viridis", ax=ax)
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
            
            # Download Processed File
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Processed CSV", csv, "sentiment_results.csv", "text/csv")
        else:
            st.error("Selected column does not contain categorical text data.")

elif input_type == "TXT File":
    st.subheader("Upload TXT File for Sentence Tokenization and Sentiment Analysis")
    txt_file = st.file_uploader("Upload a TXT file", type=["txt"])
    if txt_file:
        text_data = txt_file.read().decode("utf-8")
        sentences = tokenize_text(text_data)
        df = pd.DataFrame(sentences, columns=["Sentence"])
        df["TextBlob_Score"], df["VADER_Score"], df["Sentiment"] = zip(*df["Sentence"].apply(analyze_sentiment))
        st.dataframe(df)
        
        # Sentiment Distribution Visualization
        fig, ax = plt.subplots()
        sns.countplot(x=df["Sentiment"], palette="viridis", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
        
        # Download Processed File
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed CSV", csv, "sentiment_results.csv", "text/csv")
