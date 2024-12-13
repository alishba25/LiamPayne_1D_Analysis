#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st

st.title("A Tribute to Liam Payne")

st.markdown("""
### Welcome!
This application is a tribute to Liam Payne and his contributions to One Direction's musical journey. 
Below, you will find insights into the band's discography and Liam's unique vocal contributions.
""")

# Section 1: His Journey with One Direction
st.header("His Journey with One Direction")
st.write("Liam Payne’s journey began with *X-Factor*, where he became an essential member of One Direction.")
st.write("His voice added depth to hits like *Story of My Life* and *Night Changes*.")

# Section 2: Solo Career and Beyond
st.header("Solo Career and Beyond")
st.write("Post-1D, Liam shone as a solo artist with hits like *Strip That Down*.")
st.write("His contributions transcend music, inspiring millions worldwide.")

# Section 3: Legacy
st.header("Legacy")
st.write("Through this notebook, we celebrate Liam’s artistry, passion, and irreplaceable role in our hearts.")

st.markdown("""---
### Importing Necessary Libraries
""")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from summarytools import dfSummary
import warnings
warnings.filterwarnings('ignore')

# Streamlit code doesn't require any specific functionality; it just imports everything

# Hardcoded file path
file_path = 'C:\\Users\\User\\OneDrive\\Desktop\\New folder\\One_Direction_Proper_Dataset.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Display the dataframe
st.write(df)

# Display the dataframe
st.write("### DataFrame", df)

# Shape of the dataframe
st.write("### Shape of the DataFrame:", df.shape)

# Columns of the dataframe
st.write("### Columns of the DataFrame:", df.columns)

# First few rows of the dataframe
st.write("### First 5 rows of the DataFrame:", df.head())

# Statistical summary of the dataframe
st.write("### Statistical Summary (including all columns):", df.describe(include='all'))

# Dataframe information
st.write("### DataFrame Info:")
df_info = df.info()
st.text(df_info)

# Missing values count
st.write("### Missing Values in the DataFrame:")
st.write("NaN count:", df.isna().sum())
st.write("Null count:", df.isnull().sum())

# Duplicate rows count
st.write("### Number of Duplicate Rows:", df.duplicated().sum())

# Display the dataframe
st.write("### DataFrame", df)

# Display the summary of the dataframe using dfSummary
st.write("### DataFrame Summary")
summary = dfSummary(df)
st.write(summary)
st.markdown(summary.to_html(), unsafe_allow_html=True)

# Exploratory Data Analysis (EDA) Section
st.title("Exploratory Data Analysis (EDA)")

# Distribution of Songs by Year
st.write("### Distribution of Songs by Year")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', order=df['Year'].value_counts().index, palette='viridis')
plt.title('Distribution of Songs by Year')
plt.xticks(rotation=45)
st.pyplot(plt)

# Distribution of Songs by Album
st.write("### Distribution of Songs by Album")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Album(s)', order=df['Album(s)'].value_counts().index, palette='viridis')
plt.title("Distribution of Songs by Album")
plt.xticks(rotation=45)
st.pyplot(plt)

st.title("### Word Frequency Analysis")

# Combine all lyrics (assuming 'Lyrics' column contains the song lyrics)
all_lyrics = " ".join(df['Lyrics'].dropna())

# Count the frequency of each word
word_counts = Counter(all_lyrics.split())

# Get the 20 most common words
most_common_words = word_counts.most_common(20)

# Display the most common words
st.write("Most Common Words:", most_common_words)


# Word Cloud for Lyrics
st.write("### Word Cloud of Lyrics")

# Combine all lyrics (assuming 'Lyrics' column contains the song lyrics)
all_lyrics = " ".join(df['Lyrics'].dropna())

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_lyrics)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Lyrics')
st.pyplot(plt)

st.title("Liam Payne-Specific Data Analysis")

# Add 'is_liam_vocal' column if missing
if 'is_liam_vocal' not in df.columns:
    # Adjust logic based on your dataset
    if 'singer' in df.columns:
        df['is_liam_vocal'] = df['singer'].str.contains("Liam Payne", na=False)
    else:
        # Temporary fallback logic (if 'singer' column is missing, assume all songs are by Liam)
        df['is_liam_vocal'] = True  # or False based on your assumption

# Filter for Liam Payne's songs
liam_songs = df[df['is_liam_vocal'] == True]

# Display Liam Payne's songs in the Streamlit app
st.write("### Liam Payne's Songs")
st.write(liam_songs)

# Optional: Show the number of Liam Payne's songs
st.write(f"Total Number of Songs by Liam Payne: {len(liam_songs)}")

# Display the filtered songs (Liam Payne's songs)
st.write("### Liam Payne's Songs")
st.write(liam_songs)

# Optionally display the number of Liam Payne's songs
st.write(f"Total Number of Songs by Liam Payne: {len(liam_songs)}")

st.title("Lyrics Preprocessing")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Preprocessing Lyrics Function
def preprocess_lyrics(lyrics):
    # Remove annotations like [Verse]
    lyrics = re.sub(r"\[.*?\]", "", lyrics)
    
    # Remove special characters
    lyrics = re.sub(r"[^a-zA-Z\s]", "", lyrics)
    
    # Convert to lowercase
    lyrics = lyrics.lower()
    
    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Return cleaned lyrics
    return " ".join(tokens)

# Apply preprocessing to 'Lyrics' column
df['cleaned_lyrics'] = df['Lyrics'].apply(preprocess_lyrics)

# Add 'is_liam_vocal' column if missing
if 'is_liam_vocal' not in df.columns:
    if 'singer' in df.columns:
        df['is_liam_vocal'] = df['singer'].str.contains("Liam Payne", na=False)
    else:
        # Placeholder logic: assume all songs are by Liam
        df['is_liam_vocal'] = True  # Adjust logic as needed

# Filter for Liam Payne's songs
liam_songs = df[df['is_liam_vocal'] == True]

# Streamlit Interface
st.title("Liam Payne Song Analysis")

# Display number of Liam Payne's songs
st.write(f"Total Number of Songs by Liam Payne: {len(liam_songs)}")

# Display first few rows of filtered songs
st.write("### Liam Payne's Songs")
st.write(liam_songs[['Song', 'cleaned_lyrics']].head())

# Display cleaned lyrics for one specific song (if desired)
st.write("### Example Cleaned Lyrics of a Liam Payne Song")
example_song = liam_songs.iloc[0]  # First song
st.write(f"**Song Title:** {example_song['Song']}")
st.write(f"**Cleaned Lyrics:** {example_song['cleaned_lyrics']}")

# Optionally, you can visualize word cloud or word frequency for Liam Payne's lyrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all cleaned lyrics of Liam Payne
liam_lyrics = " ".join(liam_songs['cleaned_lyrics'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(liam_lyrics)

# Display the word cloud in Streamlit
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Liam Payne Lyrics')
st.pyplot(plt)

# Sentiment Analysis Functions
def analyze_sentiment(lyrics):
    return TextBlob(lyrics).sentiment.polarity

def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and categorization on Liam Payne's songs
liam_songs['sentiment'] = liam_songs['cleaned_lyrics'].apply(analyze_sentiment)
liam_songs['sentiment_category'] = liam_songs['sentiment'].apply(categorize_sentiment)

# Streamlit Interface
st.title("Liam Payne Song Sentiment Analysis")

# Display sentiment analysis results for Liam Payne's songs
st.write("### Liam Payne's Songs with Sentiment Analysis")
st.write(liam_songs[['Song', 'sentiment_category', 'sentiment']].head())

# Display cleaned lyrics and sentiment for one specific song
st.write("### Example Cleaned Lyrics and Sentiment of a Liam Payne Song")
example_song = liam_songs.iloc[0]  # First song
st.write(f"**Song Title:** {example_song['Song']}")
st.write(f"**Cleaned Lyrics:** {example_song['cleaned_lyrics']}")
st.write(f"**Sentiment Score:** {example_song['sentiment']}")
st.write(f"**Sentiment Category:** {example_song['sentiment_category']}")

# Visualize Sentiment Distribution
st.write("### Sentiment Distribution of Liam Payne's Lyrics")

# Create the plot
plt.figure(figsize=(10, 6))
sns.histplot(liam_songs['sentiment'], kde=True, color='purple')  # Adjusted color to use a simple one
plt.title('Sentiment Distribution of Lyrics')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(plt)

# Visualize Sentiment Category Distribution
st.write("### Sentiment Distribution in Liam Payne's Vocals")

# Create the bar plot
liam_songs['sentiment_category'].value_counts().plot(kind='bar', color=['yellow', 'lightgreen', 'red'])
plt.title("Sentiments in Liam Payne's Vocals")
plt.xlabel('Sentiment Category')
plt.ylabel('Count')

# Display the plot in Streamlit
st.pyplot(plt)

# Generate a word cloud for Liam Payne's vocal contributions
liam_lyrics = " ".join(liam_songs['cleaned_lyrics'])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(liam_lyrics)

# Display the word cloud
st.write("### Word Cloud: Liam Payne's Vocal Contributions")

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Liam Payne's Vocal Contributions")

# Show the plot in Streamlit
st.pyplot(plt)

st.title("TF-IDF Analysis of Liam Payne's Lyrics")

# TF-IDF Analysis for distinctive words in Liam Payne's lyrics
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = vectorizer.fit_transform(liam_songs['cleaned_lyrics'])
liam_unique_words = vectorizer.get_feature_names_out()

# Streamlit Interface
st.title("Identifying Distinctive Words in Liam Payne's Lyrics Using TF-IDF")

# Display the top distinctive words
st.write("### Top 20 Distinctive Words in Liam Payne's Lyrics:")
st.write(liam_unique_words)

import plotly.graph_objects as go

# Create a DataFrame with the provided data
milestones = pd.DataFrame({'Year': [2010, 2011, 2013, 2015, 2017, 2020],
                          'Event': ['Formation of One Direction on X-Factor',
                                    'First Album Release: Up All Night',
                                    'Take Me Home World Tour',
                                    'Band’s Hiatus',
                                    'Liam’s Solo Career Launch',
                                    'Liam’s Chart-Topping Singles']})

# Create a Plotly line chart with markers
fig = go.Figure(data=[go.Scatter(x=milestones['Year'], y=milestones['Event'], mode='lines+markers')])

# Set chart title and axis labels
fig.update_layout(
    title='Liam Payne’s Career Milestones',
    xaxis_title='Year',
    yaxis_title='Event',
    hovermode='x unified'  # Display hover data for all traces at the same x-coordinate
)

# Streamlit Interface
st.title("Liam Payne's Career Milestones")

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)

import plotly.express as px

# Analyze the average sentiment of Liam Payne's vocals by album
liam_sentiment_album = liam_songs.groupby('Album(s)').sentiment.mean().reset_index()

# Create the bar chart using Plotly
fig_sentiment = px.bar(
    liam_sentiment_album, x='Album(s)', y='sentiment', color='Album(s)',
    title="Average Sentiment of Liam's Vocal Contributions by Album",
    color_discrete_sequence=['yellow', 'lightgreen', 'red']
)

# Streamlit Interface
st.title("Sentiment Analysis by Album")

# Display the Plotly chart in Streamlit
st.plotly_chart(fig_sentiment)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

# Tokenizer initialization and fitting on Liam Payne's lyrics
tokenizer = Tokenizer()
liam_lyrics_list = liam_songs['cleaned_lyrics'].tolist()
tokenizer.fit_on_texts(liam_lyrics_list)

# Prepare Sequences for LSTM
sequences = tokenizer.texts_to_sequences(liam_lyrics_list)
vocab_size = len(tokenizer.word_index) + 1

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Streamlit Interface
st.title("LSTM Model for Lyric Generation")

# Show the model summary
st.write("### Model Summary")
st.text(model.summary())

# Function to generate Liam-inspired lyrics using the trained LSTM model
def generate_lyrics(model, tokenizer, seed_text, max_length=50):
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')
        predicted = model.predict(sequence, verbose=0).argmax()
        word = tokenizer.index_word.get(predicted, '')
        seed_text += ' ' + word
    return seed_text

# Streamlit Interface for generating lyrics
st.title("Liam Payne-Inspired Lyric Generator")

seed_text = st.text_input("Enter a seed text to generate lyrics:", "Through the fire")
if seed_text:
    generated_lyrics = generate_lyrics(model, tokenizer, seed_text)
    st.write("Generated lyrics (Liam-themed):")
    st.write(generated_lyrics)

# Sentiment Trend Analysis by Year
sentiment_by_year = liam_songs.groupby('Year').sentiment.mean().reset_index()

st.write("### Sentiment Trend Over Years")
fig_sentiment = plt.figure(figsize=(12, 6))
sns.lineplot(data=sentiment_by_year, x='Year', y='sentiment', marker='o', ci=None)
plt.title("Sentiment Trend Over Years", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Average Sentiment Polarity")
plt.grid()
st.pyplot(fig_sentiment)

# Sentiment Intensity Analysis
def analyze_sentiment_intensity(lyrics):
    blob = TextBlob(lyrics)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

liam_songs['sentiment_intensity'] = liam_songs['cleaned_lyrics'].apply(analyze_sentiment_intensity)
liam_songs['polarity'] = liam_songs['sentiment_intensity'].apply(lambda x: x['polarity'])
liam_songs['subjectivity'] = liam_songs['sentiment_intensity'].apply(lambda x: x['subjectivity'])

st.write("### Sentiment Intensity Analysis")
fig_intensity = plt.figure(figsize=(12, 6))
sns.scatterplot(data=liam_songs, x='polarity', y='subjectivity', hue='sentiment_category', palette='viridis')
plt.title("Sentiment Intensity Analysis", fontsize=16)
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.grid()
st.pyplot(fig_intensity)

# TF-IDF Analysis to Identify Distinctive Words
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = vectorizer.fit_transform(liam_songs['Lyrics'])
liam_unique_words = vectorizer.get_feature_names_out()

st.write("### Top 20 Distinctive Words in Liam Payne's Songs:")
st.write(liam_unique_words)

# Streamlit Interface for Conclusion
st.title("Conclusion")

st.write("""
Liam Payne’s contributions to One Direction and his solo career have been profound, shaping not only the band’s legacy but also his own as a unique artist. Through data analysis, we can uncover patterns in his vocal delivery, lyricism, and evolving fan reception. This tribute not only highlights Liam’s evolution but reveals his undeniable influence on pop music.

By diving deep into sentiment analysis, lyric frequency, and career milestones, this project showcases how Liam Payne transitioned from a key member of a global phenomenon to a chart-topping solo artist. His impact is more than just numbers; it’s a testament to the power of artistry, resilience, and transformation in the music industry.

With this data-driven journey, we honor his musical journey and the millions of fans who have supported him along the way.
""")

st.write("""
This comprehensive analysis highlighted Liam Payne's contributions and trends in One Direction's music. 
We explored lyrics, sentiments, and distinctive themes, providing insights into Liam's unique artistic journey.
""")

# In[ ]:




