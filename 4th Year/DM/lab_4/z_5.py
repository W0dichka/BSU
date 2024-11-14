import pandas as pd
import folium

gtd_data = pd.read_csv("globalterrorismdb_0718dist.csv", encoding='ISO-8859-1') 

gtd_data = gtd_data[gtd_data['latitude'].notnull() & gtd_data['longitude'].notnull()]

map_center = [gtd_data['latitude'].mean(), gtd_data['longitude'].mean()]
gtd_map = folium.Map(location=map_center, zoom_start=2)

for _, row in gtd_data.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"Событие ID: {row['eventid']}<br>Страна: {row['country_txt']}<br>Город: {row['city']}"
    ).add_to(gtd_map)

gtd_map.save("F:/4_kyrs/MIAOD/lab_4/gtd_map.html")
print("end")