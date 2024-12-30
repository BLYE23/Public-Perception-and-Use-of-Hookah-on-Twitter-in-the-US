import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load the dataset
file_path = 'total_result.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# Filter out only 'user' entries
user_data = data[data['user_or_not'] == 'not user']

# Group by week and count users
user_data.set_index('timestamp', inplace=True)
weekly_counts = user_data.resample('W').size()

# Remove the last two data points to ensure all points represent a full week
weekly_counts = weekly_counts[:-1]

# Re-plotting with updated data and less dense x-axis labels
plt.figure(figsize=(12, 6))
plt.plot(weekly_counts.index, weekly_counts, marker='o', linestyle='-')
plt.title('Number of Non-users Over Time (Weekly)')
plt.xlabel('Time')
plt.ylabel('Number of Non-users')

# Adjusting x-axis to be less dense
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Adjust interval for sparser labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True)

# Show the updated plot
plt.tight_layout()
plt.show()
