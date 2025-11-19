import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openrouteservice


from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
from flask import Flask, render_template, request, Response

client = openrouteservice.Client(key='5b3ce3597851110001cf624835110eddbce44feb9e064b60782b4a04')

geocoded_colleges_df = pd.read_csv('chicago_colleges_geoceded.csv')
print(geocoded_colleges_df)

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

print(chicago_rentals_df)

bus_routes_df = pd.read_csv("CTA_-_Bus_Routes.csv")
bus_routes_df['geometry'] = bus_routes_df['the_geom'].apply(load_wkt)

def get_closest_routes(lat, lon, bus_routes_df, top_n=3):
    point = Point(lon, lat)
    bus_routes_df['distance'] = bus_routes_df['geometry'].apply(lambda g: g.distance(point))
    return bus_routes_df.nsmallest(top_n, 'distance')

# End of Cleaned Chicago Rentals --------------------

divvy_stations_df = pd.read_csv("Divvy_Bicycle_Stations_20250405.csv")
divvy_stations_df = divvy_stations_df[['Station Name', 'Latitude', 'Longitude', 'Location']]
divvy_stations_df


def haversine(lat1, long1, lat2, long2):
    R = 3963.19  # Radius of the Earth in miles
    lat1 = np.radians(lat1)
    long1 = np.radians(long1)
    lat2 = np.radians(lat2)
    long2 = np.radians(long2)

    dlong = long2 - long1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colleges')
def colleges_json():
    return geocoded_colleges_df.to_json(orient='records')

# Lets get apartment listings for min and max price ranges
@app.route('/api/rentals', methods=['GET'])
def get_rentals():
    min_price = request.args.get('min_price', type=int, default=0)
    max_price = request.args.get('max_price', type=int, default=1_000_000)
    max_distance = request.args.get('distance', type=float, default=0)
    num_beds = request.args.get('bedrooms', type=float)
    num_bathrooms = request.args.get('bathrooms', type=float)
    prefer_distance = request.args.get('prefer-distance', default='false').lower() == 'true'
    prefer_price = request.args.get('prefer-price', default='false').lower() == 'true'
    price_priority = request.args.get('price-priority', default='distance').lower()

    university_lat = request.args.get('university_lat', type=float)
    university_lon = request.args.get('university_lon', type=float)

    university_divvy_station = find_nearest_station(university_lat, university_lon, divvy_stations_df)

    filtered_df = chicago_rentals_df[
        (chicago_rentals_df['Price'] >= min_price) &
        (chicago_rentals_df['Price'] <= max_price)
    ]

    # Calculate Distance only if Distance is selected or needed for sorting
    if prefer_distance or (prefer_distance and prefer_price):
        filtered_df['Distance'] = filtered_df.apply(
            lambda row: haversine(university_lat, university_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

    # Apply sorting logic
    if prefer_distance and prefer_price:
    # Both preferences are checked, sort by both (first by Distance, then by Price)
        filtered_df = filtered_df.sort_values(by=['Distance', 'Price'], kind='mergesort').reset_index(drop=True)
    elif prefer_distance:
        # Only distance preference is checked, sort by closest distance
        filtered_df = filtered_df.sort_values(by='Distance').reset_index(drop=True)
    elif prefer_price:
        # Only price preference is checked, sort by cheapest price
        filtered_df = filtered_df.sort_values(by='Price').reset_index(drop=True)
    else:
        # No sorting preference, just return the filtered_df as it is
        filtered_df = filtered_df.reset_index(drop=True)

    if num_beds is not None:
        filtered_df = filtered_df[filtered_df['Beds'] == num_beds]
    if num_bathrooms is not None:
        filtered_df = filtered_df[filtered_df['Baths'] == num_bathrooms]

    # filtered_df = filtered_df.head(5)

    nearby_rentals = []
    for _, row in filtered_df.iterrows():
        distance = haversine(university_lat, university_lon, row['Latitude'], row['Longitude'])
        if distance <= max_distance:
            # Haversine General Distance
            row['Distance'] = distance
            
            rental_routes = get_closest_routes(row['Latitude'], row['Longitude'], bus_routes_df)
            university_routes = get_closest_routes(university_lat, university_lon, bus_routes_df)

            # Find common routes
            direct_routes = set(rental_routes['ROUTE']).intersection(university_routes['ROUTE'])
            
            if direct_routes:
                row['Direct Bus Route'] = list(direct_routes)[0]  # You can also store all
            else:
                row['Direct Bus Route'] = None
            # Solely Walking distance
            # walking_time = get_travel_time(university_lat, university_lon, row['Latitude'], row['Longitude'])
            # row['Walking Time'] = walking_time

            # # Walk to Divvy + Bike Distance
            # rental_station = find_nearest_station(row['Latitude'], row['Longitude'], divvy_stations_df)
            # bike_walking_time = get_travel_time(row['Latitude'], row['Longitude'],
            #                                 rental_station['Latitude'], rental_station['Longitude'])
            # biking_time = get_travel_time(rental_station['Latitude'], rental_station['Longitude'],
            #                               university_divvy_station['Latitude'], university_divvy_station['Longitude'],
            #                               profile='cycling-regular')
            # if bike_walking_time is not None and biking_time is not None:
            #     total_commute_time = bike_walking_time + biking_time
            #     row['Biking Time'] = round(total_commute_time, 2)
            # else:
            #     row['Biking Time'] = None

            nearby_rentals.append(row)

    if nearby_rentals:
        result_df = pd.DataFrame(nearby_rentals)
        json_data = result_df[['Price', 'Address', 'Distance', 'Beds', 'Baths', 'Latitude', 'Longitude', 'Listing URL', 'Direct Bus Route']].to_json(orient='records')
    else:
        json_data = '[]'
    return Response(json_data, content_type='application/json')



@app.route('/api/travel-time', methods=['GET'])
def travel_time():
    lat1 = request.args.get('lat1', type=float)
    lon1 = request.args.get('lon1', type=float)
    lat2 = request.args.get('lat2', type=float)
    lon2 = request.args.get('lon2', type=float)
    mode = request.args.get('mode', default='foot-walking')

    if mode == 'foot-walking':
        time = get_travel_time(lat1, lon1, lat2, lon2, profile=mode)
        return {'travel_time': time}

    elif mode == 'divvy':
        # 1. Find nearest Divvy station near rental
        rental_station = find_nearest_station(lat1, lon1, divvy_stations_df)

        # 2. Find nearest Divvy station near university
        university_station = find_nearest_station(lat2, lon2, divvy_stations_df)

        # 3. Walk to rental divvy station
        walk_time = get_travel_time(lat1, lon1, rental_station['Latitude'], rental_station['Longitude'], profile='foot-walking')

        # 4. Bike to university divvy station
        bike_time = get_travel_time(rental_station['Latitude'], rental_station['Longitude'],
                                    university_station['Latitude'], university_station['Longitude'],
                                    profile='cycling-regular')

        if walk_time is not None and bike_time is not None:
            return {'travel_time': round(walk_time + bike_time, 2)}
        else:
            return {'travel_time': None}


# For DIVVY BIKES
def find_nearest_station(lat, lon, stations_df):
    min_dist = float('inf')
    nearest_station = None

    for _, row in stations_df.iterrows():
        dist = haversine(lat, lon, row['Latitude'], row['Longitude'])
        if dist < min_dist:
            min_dist = dist
            nearest_station = row

    return nearest_station

def get_travel_time(lat1, lon1, lat2, lon2, profile='foot-walking'):
    coords = [[lon1, lat1], [lon2, lat2]]
    try:
        routes = client.directions(coordinates=coords, profile=profile, format='geojson')
        duration = routes['features'][0]['properties']['segments'][0]['duration'] / 60
        return round(duration, 2)
    except Exception:
        return None


# -- End

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     university = None
#     if request.method == 'POST':
#         university_name = request.form['university']
#         university = next((college for college in geocoded_colleges_df.to_dict(orient='records') if college['Name'] == university_name), None)

#     data = geocoded_colleges_df.to_dict(orient='records')
#     return render_template('index.html', data=data, university=university)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with werkzerug server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutdown'

if __name__ == '__main__':
    app.run(debug=True)

