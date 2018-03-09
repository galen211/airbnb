# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 03:45:41 2018

@author: hanzhu
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import math


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from collections import Counter


os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB")

coords = pd.read_csv('latlong.csv', encoding = "ISO-8859-1")



coords['city_2'] = coords['places'].apply(lambda x: x.split("; ")[0])
coords['closest_city'] = coords['places'].apply(lambda x: x.split("; ")[-2])
coords['middle_places'] = coords['places'].apply(lambda x: x.split("; ")[1:-2])

# count number of hotels
wpt = nltk.WordPunctTokenizer()
english = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

f = lambda x : [r for r in wpt.tokenize((''.join(ch for ch in x if ch not in exclude)).lower()) if len(r) > 1]
place_types = pd.get_dummies(coords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)

ptrain = pd.concat([coords, place_types], axis=1)

############# from test set ####################################################
testcoords = pd.read_csv('latlong_test.csv', encoding = "ISO-8859-1")

testcoords['city_2'] = testcoords['places'].apply(lambda x: x.split("; ")[0])
testcoords['closest_city'] = testcoords['places'].apply(lambda x: x.split("; ")[-2])
testcoords['middle_places'] = testcoords['places'].apply(lambda x: x.split("; ")[1:-2])

f = lambda x : [r for r in wpt.tokenize((''.join(ch for ch in x if ch not in exclude)).lower()) if len(r) > 1]
place_types_test = pd.get_dummies(testcoords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)

ptest = pd.concat([testcoords, place_types_test], axis=1)
############# combine train and test sets ####################################################
place_types_all = pd.concat([ptrain, ptest])

place_types_all.to_csv('latlng_all.csv', index=False)

############# drop the following ####################################################
# locality, political, pointofinterest, establishment, neighborhood, sublocality', 
# sublocalitylevel1, 

# add premise and subpremise

place_types_all = place_types_all.drop(['locality', 'political', 'pointofinterest', 
                                        'establishment', 'neighborhood', 'sublocality', 
                                        'sublocalitylevel1'], axis=1)

place_types_all['premise_all'] = place_types_all['premise'] + place_types_all['subpremise']

place_types_all = place_types_all.drop(['premise', 'subpremise'], axis=1)

place_types_all.to_csv('latlng_edited.csv', index=False)

#def tokelist(listt):
#    joined = ' '.join(listt)
#    t = wpt.tokenize(joined.lower().strip())
#    counts = Counter(t)
#    hotels = 
#    
#    
## hotels: JW, Marriott, Ritz, Regis, Bulgari, Autograph, Gaylord, Meridien, Renaissance, 
## Tribute, seasons, mandarin, trump, Sheraton, Westin, Conrad, Waldorf, Doubletree, Curio,
## Hilton, hyatt, intercontinental, crowne, springhill, staybridge, embassym loews, omni,
## rosewood, rocco, shangri, taj, wyndham,
#    
## Inns: 4 points, aloft, towneplace, motel, ramada, home2, homewood
#    
#joined = ' '.join(test)
#t = wpt.tokenize(joined.lower().strip())
#counts = Counter(t)
#

#################################################################################

## get hotels
#coords['hotels'] = coords.apply(tokelist)
#
#import re
#f = lambda x : [r for r in re.split(r'[,;]+', re.sub(r'[^,a-z0-9]','',x.lower()).split('; |, ') if len(r) > 1]
#place_types = pd.get_dummies(coords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)
#
#import re
#f = lambda x : [r for r in re.split(r'[,;]+', re.sub(r'[^,a-z0-9]',' ',x.lower())) if len(r) > 1]
#place_types = pd.get_dummies(coords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)
#
#f = lambda x : [r for r in re.split(r"[,;[]',]+", x.lower()) if len(r) > 1]
#place_types = pd.get_dummies(coords['placetype'].map(f).apply(pd.Series).stack()).sum(level=0)
#
#re.split(r"[,;[]',]+", a.lower())
#
#set(re.split(',', re.sub(r'[^,a-z0-9]','',a.lower())))
#re.sub(r'[^,a-z0-9]',',',a.lower())
#
#set([x for x in wpt.tokenize(a) if x in english])
#
#types = ['accounting', 'airport', 'amusement_park', 
#         'aquarium', 'art_gallery', 'atm', 'bakery', 'bank', 'bar', 'beauty_salon', 
#         'bicycle_store', 'book_store', 'bowling_alley', 'bus_station', 'cafe', 
#         'campground', 'car_dealer', 'car_rental', 'car_repair', 'car_wash', 'casino', 
#         'cemetery', 'church', 'city_hall', 'clothing_store', 'convenience_store', 
#         'courthouse', 'dentist', 'department_store', 'doctor', 'electrician', 
#         'electronics_store', 'embassy', 'fire_station', 'florist', 'funeral_home', 
#         
#         ]
#[x for x in wpt.tokenize(a)]
#
#import string
#exclude = set(string.punctuation)
#s = ''.join(ch for ch in a if ch not in exclude)



