import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load and prepare the new data
data_new = pd.read_csv('tweet_commercial.csv')
data_new['tweet_timestamp'] = pd.to_datetime(data_new['tweet_timestamp'], unit='ms')
data_new.set_index('tweet_timestamp', inplace=True)

# Resample the new data to count the number of tweets per week
tweets_per_week_new = data_new.resample('W').size()

# Remove the last week
tweets_per_week_new = tweets_per_week_new[:-1]

# Plotting for the new data
plt.figure(figsize=(12, 6))
plt.plot(tweets_per_week_new.index, tweets_per_week_new.values, marker='o', linestyle='-', markersize=4)
plt.title('Number of Commercial Tweets Per Week')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.grid(True)

# Setting the x-axis format to show only year and month with a monthly interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Monthly interval
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
