import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare the data
data = pd.read_csv('total_result.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Filter out only 'user' entries
user_data = data[data['user_or_not'] == 'not user']

# Grouping the data by week and sentiment
sentiment_weekly = user_data.groupby([pd.Grouper(freq='W'), 'attitude']).size().unstack().fillna(0)

# Removing the last week to ensure all weeks have 7 days
sentiment_weekly = sentiment_weekly[:-1]

# Calculating the percentage of each sentiment per week
#sentiment_percent_weekly = sentiment_weekly.div(sentiment_weekly.sum(axis=1), axis=0) * 100

# Plotting line chart for the sentiment percentages
plt.figure(figsize=(12, 6))
plt.plot(sentiment_weekly.index, sentiment_weekly['positive'], label='Positive', color='green', marker='o')
plt.plot(sentiment_weekly.index, sentiment_weekly['negative'], label='Negative', color='red', marker='o')
plt.plot(sentiment_weekly.index, sentiment_weekly['neutral'], label='Neutral', color='gray', marker='o')
plt.title('Weekly Sentiment Count')
plt.xlabel('Week')
plt.ylabel('Number of Tweets')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()