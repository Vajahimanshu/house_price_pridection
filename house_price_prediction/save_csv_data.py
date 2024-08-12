import pandas as pd

data = {
    'location': ['Downtown', 'Suburb', 'City Center', 'Rural', 'Downtown', 'Suburb', 'City Center', 'Rural'],
    'size': [1200, 1500, 900, 2000, 1100, 1400, 950, 2100],
    'bedrooms': [3, 4, 2, 5, 2, 3, 2, 4],
    'bathrooms': [2, 3, 1, 3, 2, 2, 1, 3],
    'amenities': ['Pool,Gym', 'Garden,Garage', 'Pool,Gym', 'Garden,Garage', 'Pool,Gym', 'Garden,Garage', 'Pool,Gym', 'Garden,Garage'],
    'price': [500000, 600000, 450000, 550000, 480000, 520000, 460000, 570000]
}

df = pd.DataFrame(data)
df.to_csv('house_prices.csv', index=False)
