# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:39:52 2018

@author: hanzhu
"""

from googleplaces import GooglePlaces, types, lang

API_KEY = 'AIzaSyBzZBDviJmJj8zeuKH7ykBmQZIb8GSozS4'

google_places = GooglePlaces(API_KEY)

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import math
from scipy import stats
import seaborn as sns
import datetime

import dill

os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB")
# Adjust screen output size
#pd.util.terminal.get_terminal_size() # get current size
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['dataset'] = "train"
test['dataset'] = "test"
data = pd.concat([train,test], axis = 0).reset_index()



query_result = google_places.nearby_search(
        lat_lng={'lat': train['longitude'][0], 'lng': 5.train['latitude'][0]},
        radius = 20000,
        type = ['bar'])

query_result = google_places.nearby_search(
        lat_lng={'lat': 38.863534, 'lng': -76.949223},
        radius = 20000)

query_result = google_places.nearby_search(
        lat_lng={'lat': 38.863534, 'lng': -76.949223},
        radius = 10000)


query_result = google_places.nearby_search(
        lat_lng={'lat': 38.863534, 'lng': -76.949223})

Washington
Anacostia Railroad Bridge
The Shops at Iverson
Budget Inn
The Seed School
Stanton Elementary School
Temple Hills Skate Palace
Diverse Web Design
KIPP DC Benning Campus
Suitland Federal Center
Iverson Mall Merchants Association
Dr. Yolandra E. Hancock, MD
SUBWAY®Restaurants
Suitland High School
Forman Mills
Capital Crossing Apartments
Fort Dupont Ice Arena
Anacostia High School
Kramer Middle School
Northeast Washington

for place in query_result.places: print(place.vicinity)

details = []
for place in query_result.places: 
    details.append(place.details)


##############################################################
    
latlong = train[['id', 'dataset', 'latitude', 'longitude']]
latlong.insert(3, 'places', '')
latlong.insert(4, 'placetype', '')
latlong['lat_round'] = round(latlong['latitude'], 3)
latlong['lng_round'] = round(latlong['longitude'], 3)
latlong['latlng'] = latlong['lat_round'].map(str) + ','+ latlong['lng_round'].map(str)

latlong2 = test[['id', 'dataset', 'latitude', 'longitude']]
latlong2.insert(3, 'places', '')
latlong2.insert(4, 'placetype', '')
latlong2['lat_round'] = round(latlong2['latitude'], 3)
latlong2['lng_round'] = round(latlong2['longitude'], 3)
latlong2['latlng'] = latlong2['lat_round'].map(str) + ','+ latlong2['lng_round'].map(str)
# how many unique latlng pairs

z= []
for x in latlong2['latlng'].unique():
    if x not in latlong['latlng'].unique():
        z.append(x)

len(latlong2['latlng'].unique()) #47755
#39617
#18664

c = pd.DataFrame(latlong['latlng'].unique(), columns=['latlngcoords'])
c['lat'] = c['latlngcoords'].apply(lambda x: float(x.split(",")[0]))
c['lng'] = c['latlngcoords'].apply(lambda x: float(x.split(",")[1]))
c['name'] = ''
c['place_id'] = ''
c['types'] = ''
c['places'] = ''
c['place_types'] = ''

from itertools import *
place_dict = {}

for index, row in islice(c.iterrows(), 158, None):
    print(row['places'])

spot = c.shape[0]-c[c['places']==''].shape[0]
c['places'][2880:2883]

i = index

for index, row in islice(c.iterrows(), 38641, None):
    query_result = google_places.nearby_search(
        lat_lng={'lat': row['lat'], 'lng': row['lng']},
        radius = 5000)
    places = ''
    place_type = ''
    for place in query_result.places:
        places += (str(place.name)+'; ')
        place_type += (str(place.types)+'; ')
     #   place_dict[str(place)] = {}
#        place_dict[str(place)]['name'] = place.name
#        place_dict[str(place)]['place_id'] = place.place_id
#        place_dict[str(place)]['geo_location'] = place.geo_location
#        place_dict[str(place)]['types'] = place.types
    c.set_value(index, 'places', places)
    c.set_value(index, 'placetype', place_type)
    print(places[0:10])
    
c.to_csv('latlong2_test.csv', index=False)

#7:28

c = pd.DataFrame(z, columns=['latlngcoords'])
c['lat'] = c['latlngcoords'].apply(lambda x: float(x.split(",")[0]))
c['lng'] = c['latlngcoords'].apply(lambda x: float(x.split(",")[1]))
c['places'] = ''
c['place_type'] = ''

from itertools import *
place_dict = {}

for index, row in islice(c.iterrows(), 158, None):
    print(row['places'])

spot = c.shape[0]-c[c['places']==''].shape[0]
c['places'][2880:2883]

i = index

for index, row in islice(c.iterrows(), 7045, None):
    query_result = google_places.nearby_search(
        lat_lng={'lat': row['lat'], 'lng': row['lng']},
        radius = 5000)
    places = ''
    place_type = ''
    for place in query_result.places:
        places += (str(place.name)+'; ')
        place_type += (str(place.types)+'; ')
     #   place_dict[str(place)] = {}
#        place_dict[str(place)]['name'] = place.name
#        place_dict[str(place)]['place_id'] = place.place_id
#        place_dict[str(place)]['geo_location'] = place.geo_location
#        place_dict[str(place)]['types'] = place.types
    c.set_value(index, 'places', places)
    c.set_value(index, 'placetype', place_type)
    print(places[0:10])
    
c.to_csv('latlong_test.csv', index=False)