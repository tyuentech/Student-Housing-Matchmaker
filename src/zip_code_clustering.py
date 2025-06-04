import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')

# Cleaned Chicago Rentals ------------------
chicago_rentals_df = pd.read_csv("Zillow Rental (Chicago Area 03_03_2025).csv")
chicago_rentals_df = chicago_rentals_df[['Price', 'Address', 'Listing URL', 'Short Address', 'Zip', 'Beds', 'Baths', 'Raw Property Details', 'Latitude', 'Longitude']].reset_index(drop=True)

# Keep Beds and Baths as floats, cleaned NaN baths
chicago_rentals_df['Baths'] = chicago_rentals_df['Baths'].fillna(1)
chicago_rentals_df['Beds'] = chicago_rentals_df['Beds'].astype(float)

# Dropped undisclosed Address from database so left with 2731 rows.
chicago_rentals_df = chicago_rentals_df[chicago_rentals_df['Short Address'] != '(undisclosed Address)'].reset_index(drop=True)

# Cleaned Price column to show as type int
chicago_rentals_df['Price'] = (
    chicago_rentals_df['Price']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

chicago_rentals_df['Price'] = pd.to_numeric(chicago_rentals_df['Price'], errors='coerce')
chicago_rentals_df['Price'] = chicago_rentals_df['Price'].astype(float)

# End of Cleaned Chicago Rentals --------------------

# Group by Zip code, compute average price
zip_prices = chicago_rentals_df.groupby('Zip')['Price'].mean().reset_index()

# Arrange Prices around cluster's center of gravity as we learned in lecture
scaler = StandardScaler()
zip_prices['Scaled Price'] = scaler.fit_transform(zip_prices[['Price']])

# Running K means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
zip_prices['Cluster'] = kmeans.fit_predict(zip_prices[['Scaled Price']])

print(zip_prices.groupby('Cluster')['Price'].mean())
clusters = {
    0: 'Low Average',
    1: 'Mid Average',
    2: 'High Average'
}

zip_prices['Cluster'] = zip_prices['Cluster'].map(clusters)

# Plot
plt.figure(figsize=(13, 6))
sns.barplot(x='Zip', y='Price', hue='Cluster', hue_order=['Low Average', 'Mid Average', 'High Average'], data=zip_prices, palette=sns.color_palette('Purples', n_colors=3))

plt.title('Zip Codes Clustered by Average Price')

# Make Zip Codes viewable
plt.xticks(rotation=45)
plt.xlabel('Zip Code')

plt.ylabel('Average Listing Price')

plt.legend(title='Clusters', bbox_to_anchor=(1.20, 1), loc='upper right')

plt.tight_layout()
plt.show()