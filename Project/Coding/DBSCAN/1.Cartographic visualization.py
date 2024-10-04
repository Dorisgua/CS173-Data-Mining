import pandas as pd
import folium


data = pd.read_csv("../data/GTDdata200.csv", encoding='latin-1')

data['iyear'] = data['iyear'].astype(int)

for year in range(2011, 2022):
    year_data = data[data['iyear'] == year]

    year_data = year_data.dropna(subset=['latitude', 'longitude'])

    map_obj = folium.Map(location=[0, 0], zoom_start=2)

    for idx, row in year_data.iterrows():
        popup_content = f"{row['iyear']}/{row['imonth']}/{row['iday']}|{row['gname']}"
        popup_content = popup_content.replace('nan', 'unknown')
        folium.Marker([row['latitude'], row['longitude']], popup=popup_content).add_to(map_obj)

    map_obj.save(f"map{year}.html")
