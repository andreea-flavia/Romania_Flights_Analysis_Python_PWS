# Linii pentru prelucrare date

import pandas as pd
df = pd.read_csv('europe_flights_google_prices.csv')
columns = df.columns;
print(columns)
df_ro = df[(df['source_country'] == 'Romania') | (df['destination_country'] == 'Romania')].copy()
df_ro = df_ro[~df_ro['airline'].str.contains(r'\+', na=False)].copy()
df_ro.to_csv('flights_ro.csv', index=False)
#print(df_ro.head())
df_ro['departure_time'] = pd.to_datetime(df_ro['departure_time'])
df_ro['arrival_time'] = pd.to_datetime(df_ro['arrival_time'])
df_ro['scraped_at'] = pd.to_datetime(df_ro['scraped_at'], errors='coerce')

dur = df_ro['duration'].str.extract(r'(?P<hours>\d+)h\s+(?P<minutes>\d+)m').astype(int)
df_ro['duration_minutes'] = dur['hours'] * 60 + dur['minutes']

df_ro['route'] = df_ro['source_airport'] + ' -> ' + df_ro['destination_airport']
df_ro['country_pair'] = df_ro['source_country'] + ' -> ' + df_ro['destination_country']
df_ro['is_nonstop'] = df_ro['stops'] == 0

df_ro.to_csv('flights_ro_details.csv', index=False)