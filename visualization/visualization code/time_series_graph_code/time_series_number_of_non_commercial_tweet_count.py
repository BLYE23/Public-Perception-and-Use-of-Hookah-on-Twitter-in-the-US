import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load and prepare the data
data = pd.read_csv('total_result.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Resample the data to count the number of tweets per week
tweets_per_week = data.resample('W').size()

# Remove the last week because it is incomplete
tweets_per_week = tweets_per_week[:-1]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(tweets_per_week.index, tweets_per_week.values, marker='o', linestyle='-', markersize=4)
plt.title('Number of Non-commercial Tweets Per Week')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.grid(True)

# Setting the x-axis format to show only year and month with a larger interval
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Monthly interval
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
