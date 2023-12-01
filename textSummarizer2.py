# importing libraries
import nltk 
import pandas as pd
import streamlit as st
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

stopWords = set(stopwords.words("english"))


def run_model(text):
    # Tokenizing the text

    words = word_tokenize(text)
    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    st.subheader(" Summary generated:")
    st.write(summary)
    return summary


# Input text - to summarize

def overall_sent(value):
    # decide sentiment as positive, negative and neutral
    if value >= 0.05:
        return("Positive")

    elif value <= - 0.05:
        return("Negative")

    else:
        return("Neutral")


def sentiment_analysis(text):
    # create a sentimentAnalyzer object
    obj = SentimentIntensityAnalyzer()  # to give intensity scores to sentences
    # Polarity scores - Return a float for sentiment strength based on the input text. Positive values are positive valence, negative value are negative valence.
    sentiment_dict = obj.polarity_scores(text)
    outputRes = "Sentence was rated as " + str(sentiment_dict['neg']*100) + "% Negative, " + str(sentiment_dict['neu']*100) + "% Neutral, " + str(
        sentiment_dict['pos']*100) + "% Positive" + " and " + str(sentiment_dict['compound']*100) + "% Compound"
    st.write(outputRes)
    overall_rating = overall_sent(sentiment_dict['compound'])
    # compound score is normalized between 0 and 1 and is the sum of positive (closer to 1), negative and neutral scores.
    st.write("Sentence Overall Rated As", overall_rating)


if __name__ == "__main__":
    st.title("Text summarizer application")
    text = st.text_area("Enter the text that need to be summarized")
    if st.button('Submit'):
        summary = run_model(text)
        st.title("Sentiment Analysis of Summary")
        sentiment_analysis(summary)
        st.title("Sentiment Analysis of Text")
        sentiment_analysis(text)
