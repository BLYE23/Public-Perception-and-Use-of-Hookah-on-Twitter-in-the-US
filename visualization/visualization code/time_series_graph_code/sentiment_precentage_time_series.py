import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare the data
data_sentiment = pd.read_csv('total_result.csv')
data_sentiment['timestamp'] = pd.to_datetime(data_sentiment['timestamp'], unit='ms')
data_sentiment.set_index('timestamp', inplace=True)

# Grouping the data by week and sentiment
sentiment_weekly = data_sentiment.groupby([pd.Grouper(freq='W'), 'attitude']).size().unstack().fillna(0)

# Removing the last week to ensure all weeks have 7 days
sentiment_weekly = sentiment_weekly[:-1]

# Calculating the percentage of each sentiment per week
sentiment_percent_weekly = sentiment_weekly.div(sentiment_weekly.sum(axis=1), axis=0) * 100

# Plotting line chart for the sentiment percentages
plt.figure(figsize=(12, 6))
plt.plot(sentiment_percent_weekly.index, sentiment_percent_weekly['positive'], label='Positive', color='green', marker='o')
plt.plot(sentiment_percent_weekly.index, sentiment_percent_weekly['negative'], label='Negative', color='red', marker='o')
plt.plot(sentiment_percent_weekly.index, sentiment_percent_weekly['neutral'], label='Neutral', color='gray', marker='o')
plt.title('Weekly Sentiment Proportions')
plt.xlabel('Week')
plt.ylabel('Percentage of Tweets (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
