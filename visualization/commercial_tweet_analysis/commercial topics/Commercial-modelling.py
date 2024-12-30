#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install bertopic')


# In[5]:


from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import operator
import string
import nltk
import re

from bertopic import BERTopic

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[141]:


# Read JSON file
comme = pd.read_json('tweet_commercial.json', lines = True)

comme.head()


# In[142]:


# lemmatizer.py
lemmatizer = WordNetLemmatizer()

##Tags the words in the tweets
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:
        return(None)

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))


# In[143]:


# Define the languages
languages = [
    'english',  # English
    'dutch',  # Netherlands, Belgium
    'french', # Belgium
    'german', # Belgium
    'danish',  # Denmark
    'swedish',  # Sweden
    'norwegian',  # Norway
]

# Gather stopwords for all the languages
stop_words = set()
for lang in languages:
    stop_words.update(stopwords.words(lang))

def tweet_cleaner(tweet):
    
    #remove all links (starting with http)
    tweet = re.sub(r'http\S+', '', tweet)

    #remove emojis and all the str beginning with '\'
    tweet = re.sub(r'[\U00010000-\U0010ffff]', '', tweet)

    #remove '@name'
    tweet = re.sub(r'(@.*?)[\s]', ' ', tweet)

    #replace '&amp;' with '&'
    tweet = re.sub(r'&amp;', '&', tweet)
    
    #remove numbers
    tweet = re.sub(r'[0-9]', '', tweet)

    #remove stopwords
    words = word_tokenize(tweet)
    words = [w for w in words if w.lower() not in stop_words]

    #remove punctuation
    words = [''.join(char for char in w if char not in string.punctuation) for w in words]
    

    #remove all words that are shorter than 3 characters
    words = [w for w in words if len(w) > 3]
    
    return ' '.join(words)

comme['text_clean']  = comme['text'].apply(tweet_cleaner)


# In[145]:


#bertopic
model = BERTopic()
topics, _ = model.fit_transform(comme['text_clean'])


# In[186]:


top_5_topics = []

for i in range(5):
    topic_words = model.get_topic(i-1)
    top_words = [word for word, _ in topic_words[:10]]
    top_5_topics.append(top_words)

# Print the top 5 topics and top 10 words for each topic
for i, topic in enumerate(top_5_topics, 1):
    print(f'Topic {i}: {", ".join(topic)}')


# In[160]:


import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic


df = comme
df['topic'] = topics

# Group by week and topic, count the number of tweets, and select the top 5 topics
weekly_topic_counts = df.groupby([pd.Grouper(key='created_at', freq='W-Mon'), 'topic']).size().unstack(fill_value=0)
top_5_topics = weekly_topic_counts.sum().nlargest(5).index
top_5_weekly_topic_counts = weekly_topic_counts[top_5_topics]

# Plot the top 5 topics over time
plt.figure(figsize=(12, 8))

for topic in top_5_weekly_topic_counts.columns:
    plt.plot(top_5_weekly_topic_counts.index, top_5_weekly_topic_counts[topic], label=f'Topic {topic+2}')

plt.xlabel('Date(Weekly)')
plt.ylabel('Number of Topics Mentioned in Tweet')
plt.title('Top 5 Topic Distribution Over Time(Weekly)')
plt.legend()
plt.grid()
plt.show()


# In[199]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

cleaned_texts = comme['text_clean']

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Fit LDA model
num_topics = 5 
lda = LatentDirichletAllocation(n_components=num_topics)
lda.fit(X)
topic_keywords = []
for topic_idx, topic in enumerate(lda.components_):
    top_keywords_idx = topic.argsort()[:-11:-1]
    top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_keywords_idx]
    topic_keywords.append(", ".join(top_keywords))
for i in range(5):
    print(f"Topic {i+1}: {topic_keywords[i]}")


# In[163]:


df2 = comme

topic_lda = lda.fit_transform(X)
df2['topic'] = topic_lda.argmax(axis=1)

df2['created_at'] = pd.to_datetime(df2['created_at'])


topic_counts = df2.groupby([pd.Grouper(key='created_at', freq='W-Mon'), 'topic']).size().unstack(fill_value=0)
# Plot the topics over time
plt.figure(figsize=(12, 8))

for topic in topic_counts.columns:
    plt.plot(topic_counts.index, topic_counts[topic], label=f'Topic {topic+1}')

plt.xlabel('Date(Weekly)')
plt.ylabel('Number of Topics Mentioned in Tweet')
plt.title('Top 5 Topic Distribution Over Time(Weekly)')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




