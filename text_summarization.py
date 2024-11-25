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

# Example usage
if __name__ == "__main__":
    # Example text
    text = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence. 
    It deals with the interaction between computers and humans using natural language. 
    NLP enables machines to understand, interpret, and generate human language. 
    Applications of NLP include sentiment analysis, chatbots, and machine translation.
    """

    # Initialize the summarizer
    summarizer = TextSummarizer()

    # Extractive Summarization
    print("Extractive Summary:")
    print(summarizer.summarize(text, method='extractive', num_sentences=2))

    # Simple Abstractive Summarization
    print("\nSimple Abstractive Summary:")
    print(summarizer.summarize(text, method='abstractive', num_sentences=3))

    # Interactive Input
    print("\nInteractive Mode:")
    user_text = input("Enter text to summarize: ")
    user_method = input("Choose summarization method ('extractive' or 'abstractive'): ").strip().lower()
    if user_method not in ['extractive', 'abstractive']:
        print("Invalid method. Defaulting to abstractive summarization.")
        user_method = 'abstractive'

    # Perform summarization based on user input
    summary = summarizer.summarize(user_text, method=user_method, num_sentences=3)
    print("\nSummary:\n", summary)
