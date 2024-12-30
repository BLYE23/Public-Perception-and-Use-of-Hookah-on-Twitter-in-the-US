import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Load and prepare the data
data_new = pd.read_csv('tweet_hookah_filtered.csv')
data_new['tweet_timestamp'] = pd.to_datetime(data_new['tweet_timestamp'], unit='ms')
data_new.set_index('tweet_timestamp', inplace=True)

# Counting the unique users per week
unique_users_per_week = data_new.groupby(pd.Grouper(freq='W'))['tweet_user_id_str'].nunique()

unique_users_per_week = unique_users_per_week[:-1]


# Plotting for the unique users per week
plt.figure(figsize=(12, 6))
plt.plot(unique_users_per_week.index, unique_users_per_week.values, marker='o', linestyle='-', markersize=4)
plt.title('Number of Unique Users Per Week')
plt.xlabel('Date')
plt.ylabel('Number of Unique Users')
plt.grid(True)

# Setting the same x-axis format as before
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Monthly interval
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
