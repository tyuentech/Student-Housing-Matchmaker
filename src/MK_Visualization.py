import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')


chicago_rentals_df = pd.read_csv("Zillow Rental (Chicago Area 03_03_2025).csv")

chicago_rentals_df['Price'] = (
    chicago_rentals_df['Price']
    .astype(str)
    .str.replace(r'[$,]', '', regex=True)
    .astype(float)
)

price_bins = [0, 1000, 1500, 2000, 2500, 3000, float('inf')]
price_labels = ['$0–1000', '$1000–1500', '$1500–2000', '$2000–2500', '$2500–3000', '$3000+']
chicago_rentals_df['Price Range'] = pd.cut(chicago_rentals_df['Price'], bins=price_bins, labels=price_labels)

# Plotting the map
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=chicago_rentals_df,
    x='Longitude',
    y='Latitude',
    hue='Price Range',
    palette='Spectral',
    alpha=0.7
)
plt.title("Chicago Rental Listings by Price Range")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Price Range", loc="upper right")
plt.tight_layout()
plt.show()