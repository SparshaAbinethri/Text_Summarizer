import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Define the TextSummarizer class
class TextSummarizer:
    def __init__(self):
        # We won't use transformers, so no model loading here.
        pass

    def preprocess_text(self, text):
        # Basic text cleaning
        return text.strip()

    def extractive_summary(self, text, num_sentences=3):
        """
        Perform extractive summarization by selecting the most relevant sentences.
        """
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return "Text is too short to summarize."

        # Create a TF-IDF vectorizer and fit on the sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute scores for each sentence
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Rank the sentences by their scores
        ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
        return ' '.join(ranked_sentences)

    def simple_abstractive_summary(self, text, num_sentences=3):
        """
        A basic approach to abstractive summarization by selecting important sentences.
        """
        # Here, we don't use advanced methods like transformers.
        # Instead, we will pick key sentences using TF-IDF scores.
        
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return "Text is too short to summarize."

        # Create a TF-IDF vectorizer and fit on the sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute scores for each sentence
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Rank the sentences by their scores
        ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
        
        # Join the ranked sentences to form the summary (basic form of abstraction)
        return ' '.join(ranked_sentences)

    def summarize(self, text, method='abstractive', **kwargs):
        """
        Choose between extractive and simple abstractive summarization.
        """
        # Preprocess the input text
        text = self.preprocess_text(text)

        # Perform the chosen summarization method
        if method == 'extractive':
            return self.extractive_summary(text, **kwargs)
        elif method == 'abstractive':
            return self.simple_abstractive_summary(text, **kwargs)
        else:
            raise ValueError("Invalid method! Choose 'extractive' or 'abstractive'.")

# Streamlit app interface
def main():
    st.title("Text Summarizer")
    
    # Input text area
    text_input = st.text_area("Enter text for summarization:", height=200)

    if text_input:
        # Choose method
        method = st.radio("Choose summarization method:", ("extractive", "abstractive"))
        
        # Set number of sentences for extractive or abstractive summarization
        num_sentences = st.slider("Select number of sentences in summary:", 1, 5, 3)
        
        # Initialize the summarizer
        summarizer = TextSummarizer()
        
        # Summarize based on method
        if method == "extractive":
            summary = summarizer.summarize(text_input, method='extractive', num_sentences=num_sentences)
        elif method == "abstractive":
            summary = summarizer.summarize(text_input, method='abstractive', num_sentences=num_sentences)

        # Display summary
        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
