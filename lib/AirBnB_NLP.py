# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:49:40 2018

@author: hanzhu
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import math
from scipy import stats
import seaborn as sns
import datetime

import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import itertools
import re
#os.chdir("C:\Users\hanzhu\Documents\DAT210x-master\Ames House Prices")
os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB")


# Adjust screen output size
#pd.util.terminal.get_terminal_size() # get current size
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = pd.concat([train,test], axis = 0).reset_index()

desc = data[['id', 'description']]

### Text Cleaning Test ###

wpt = nltk.WordPunctTokenizer()
english = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

a = desc['description']
b = re.sub(r'[^a-zA-Z\s]', '', a, re.I|re.A)
c = b.lower().strip()
d = wpt.tokenize(c)

e = np.vectorize(d)



### Text Cleaning Real ###

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def normalize_text(text):
    # re.sub(r'[^a-zA-Z\s]', '', a, re.I|re.A) --> delete all punctuation and
    # non-character letters
    # strip = strip of trailing and leading spaces
    # lower = make all lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower().strip()
    # tokenize
    tokens = wpt.tokenize(text)
    # filter stopwords out of document
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                       if ((token not in stop_words) & (token in english))]
    # re-create document from filtered tokens
    text = ' '.join(filtered_tokens)
    return text

normalize_corpus = np.vectorize(normalize_text)

desc['normtext'] = [normalize_text(x) for x in desc['description']]


### Bag of Words ###

# Change normtext to a list of strings
tlist = []
for i in desc['normtext']:
    tlist.append(i)
    
allwords = ' '.join(tlist)

a = set(nltk.word_tokenize(allwords))
a = sorted(a) #14,439 unique words

# What is the distribution of words?
diction = {}
for entry in tlist:
    for word in wpt.tokenize(entry):
        n = entry.count(word)
        if word in diction:
            diction[word] += n
        else:
            diction[word] = n

import operator
sortedx = sorted(diction.items(), key=operator.itemgetter(1), reverse=True)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0., max_df=1., max_features=11000)
cv_matrix1 = cv.fit_transform(tlist)
cv_matrix1 = cv_matrix1.toarray()
headers = np.array(cv.get_feature_names())

# concat headers and matrix
words = pd.DataFrame(cv_matrix1, columns=headers)
words.to_csv('wordmatrix.csv', index=False)

# get the words that matter
kitchen = ['kitchen', 'kitchenette', 'cooking', 'cook', 'oven']
view = ['view', 'panoramic']
walk = ['walk', 'walkable', 'walking']
positive = ['great', 'good']
pos_sent = ['inviting', 'colorful', 'gold', 'sensitive', 'desirable', 'plush', 'romantic', 'adorable', 'tastefully','tasteful',
            'cute', 'nicely', 'pleasure', 'favorite', 'quintessential', 'pretty', 'interesting', 'lovely', 'enjoy', 'enjoying']
very_pos = ['best', 'perfect', 'perfectly', 'amazing', 'super', 'wonderful', 'grand', 'awesome', 'excellent', 
            'fantastic', 'incredible', 'spectacular', 'fabulous', 'gem', 'authentic', 'rare', 'exciting', 
            'incredibly', 'rich', 'dream', 'magnificent', 'deluxe', 'scenic', 'picturesque', 'exceptional', 'paradise']
new = ['new', 'newly']
big = ['big', 'large', 'huge', 'giant', 'massive', 'sweeping', 'enormous', 'expansive', 'roomy']
near = ['near', 'close', 'nearby', 'proximity', 'adjacent']
minutes = ['minute', 'min']
quiet = ['quiet', 'peaceful', 'serene', 'tranquil', 'calm', 'peace']
beautiful = ['beautiful', 'beautifully', 'gorgeous', 'beauty']
city = ['city', 'urban', 'metropolitan']
transport = ['transport', 'train', 'subway', 'bus', 'transportation', 'transit', 'commute', 'rail', 'railroad', 'shuttle']
comfortable = ['comfortable', 'comfy', 'comfortably', 'comfort']
cozy = ['cozy', 'coziness', 'homey', 'cosy']
central = ['central', 'center', 'centrally']
dine = ['dine', 'dinner', 'eating', 'gourmet', 'delicious', 'cuisine', 'dining', 'food', 'restaurant']
art = ['art', 'artist', 'artistic']
many = ['many', 'numerous', 'ample', 'plenty']
shopping = ['shopping', 'shop', 'store', 'plaza', 'bakery']
porch = ['porch', 'patio', 'deck']
entire = ['entire', 'whole']
couch = ['couch', 'sofa']
historic = ['historic', 'history', 'historical']
outdoors = ['outdoors', 'mountain', 'outdoor', 'lush', 'trail', 'nature', 'glen', 'forest', 'valley', 'apple', 'oak',
            'sanctuary', 'bike', 'sunset', 'natural']
sunny = ['sunny', 'bright', 'sun', 'sunlight', 'sunshine', 'sunlit']
easy = ['easy', 'convenient', 'convenience', 'conveniently']
attractions = ['museum', 'theater', 'entertainment', 'entertaining', 'movie', 'botanical', 'rodeo', 'observatory', 'botanic',
               'dance', 'landmark', 'aquarium', 'monument']
market = ['market', 'supermarket', 'grocery']
luxury = ['luxury', 'luxurious', 'penthouse', 'oasis', 'gated', 'resort', 'villa', 'estate', 'mansion']
spa = ['spa', 'sauna']
charming = ['charming', 'charm']
near_ocean = ['ocean', 'pacific', 'bay', 'marina', 'pier', 'waterfront', 'sand', 'harbor', 'wharf', 'boat', 'shore', 
              'seaport', 'alcove', 'coast', 'surf']
near_small_water = ['river', 'lake', 'ferry', 'riverside', 'pond', 'creek', 'canal']
famous = ['famous', 'fame', 'iconic']
relax = ['relax','relaxed', 'relaxation', 'unwind']
university = ['university', 'college', 'student']
fast = ['fast', 'quick', 'quickly']
fitness = ['fitness', 'yoga', 'gym']
modern = ['modern', 'contemporary']
vacation = ['vacation', 'retreat']
busy = ['busy', 'bustling', 'bustle', 'lively', 'hustle']
stone = ['stone', 'marble', 'granite']
upscale = ['upscale', 'elegant', 'chic', 'exclusive', 'premium', 'prestigious', 'fancy', 'expensive', 'vanity', 'fashion']
architecture = ['architecture', 'architectural']
vintage = ['vintage', 'antique']
culture = ['culture', 'cultural']
fresh = ['fresh', 'freshly']
sport = ['sport', 'stadium', 'golf', 'basketball', 'tennis']
bar = ['bar', 'wine', 'beer']
music = ['music', 'jazz']
safe = ['safe', 'secure', 'security']
smoke = ['smoke', 'smoking']
traffic = ['traffic', 'highway']
pet = ['pet', 'dog']
basic = ['basic', 'simple']
church = ['church', 'cathedral']
cheap = ['cheap', 'discount', 'bonus']
pool = ['pool', 'swimming']

lists = [kitchen, view, walk, positive, pos_sent, very_pos, new, big, near, minutes, quiet, beautiful, 
         city, transport, comfortable, cozy, central, dine, art, many, shopping, porch, entire, couch,
         historic, outdoors, sunny, easy, attractions, market, luxury, spa, charming, near_ocean, 
         near_small_water, famous, relax, university, fast, fitness, modern, vacation, busy, stone, upscale,
         architecture, vintage, culture, fresh, sport, bar, music, safe, smoke, traffic, pet, basic,
         church, cheap, pool]

words2 = pd.DataFrame()

for item in lists:
    x = 1
    words2[item[0]] = words[item[0]]
    while x < len(item):
        words2[item[0]] += words[item[x]]
        x += 1

# rename some columns
words2 = words2.rename(columns={'great':'positive', 'inviting':'pos_sent', 
                                'best':'very_pos', 'museum':'attractions', 'ocean':'near_ocean',
                                'river':'near_small_water'})
    


keep_words = ['private', 'home', 'house', 'full', 'park', 'living',
              'space', 'away', 'access', 'two', 
              'neighborhood', 'area', 'floor', 'beach', 
              'parking', 'queen',
              'location', 'free', 'complimentary', 'distance', 'spacious', 'clean',
              'downtown', 'studio', 'need', 'square',
              'coffee', 'well', 'fully', 'heart', 'pizza',
              'station', 'love', 
              'garden', 'everything', 'light', 'small', 'high', 'west', 
              'table', 'air',
              'closet', 'furnished', 'village', 'open', 'nice', 'stop', 
              'front', 'short', 
              'public', 'top', 'friendly', 'work', 
              'king', 'separate', 
              'suite', 'solo',  'balcony', 'local',
              'privacy', 'basement', 'airport', 'share',
              'little', 'old', 'south', 'phone', 'hidden', 'major', 'happy',
              'experience', 'north', 'tub', 'additional',
              'style', 'stylish', 'hotel', 'breakfast',
              'garage', 'furniture', 'upper', 'lower', 'cool',
              'tree', 'unique', 'fun', 'brownstone', 'refrigerator', 'community', 'hot',
              'double', 'ideal', 'half',
              'duplex', 'fireplace', 'rental', 'office', 'lounge', 
              'circle', 'decorated', 'bridge', 'accessible', 'soho',
              'terrace', 'accommodate', 'national', 'ill', 'golden', 'complete', 'cottage',
              'island',
              'fort', 'lax', 'warm', 'vibrant', 'classic', 'explore', 'dresser', 
              'smart', 'doorman', 'toaster', 'broadway', 
              'glass', 'summer', 'canyon', 'bungalow', 'extremely',
              'anywhere', 'roommate', 'designed',
              'airy', 'grove', 'popular', 'facing', 'courtyard', 'library',
              'noise', 'used', 'echo', 'personal', 'however', 'cleaning',
              'young',
              'special',
              'diverse', 'field', 'century',
              'quaint', 'hospital', 'bunk',
              'organic', 'far', 'burbank',
              'boardwalk', 'eclectic', 'variety', 'club',
              'sweet', 'concierge', 'escape',
              'connected', 
              'chill', 'wooden',
              'party',
              'flexible', 'boulevard', 'nook', 'energy',
              'professionally',
              'gallery', 'vista', 
              'skylight', 'suitable', 'important', 'communal', 'cabin',
              'game',
              'tour', 'social', 'breeze',
              'quarter', 'rustic', 'hammock', 'detached', 'alone', 'empty', 
              'rented', 'freedom',
              'traditional', 'soft', 'standard', 
              'weekly', 'monthly', 'uptown',
              'treat', 'underground', 
              'sorry', 'barbecue', 'parlor',
              'finished', 'foggy',
              'retail', 'remote',
              'occasionally']

for i in keep_words:
    words2[i] = words[i]
    
# save words2
words2.to_csv('words_2.csv', index=False)

# combine with original data
data_desc = pd.concat([data, words2], axis=1)

data = data.drop(['id', 'description', 'latitude', 'longitude', 'name',
                  'thumbnail_url'], axis=1)


'''Property Type'''

data['property_type'] = data['property_type'].replace(['Bed & Breakfast', 'Bungalow', 'Villa', 'Guest suite'], 'Guesthouse')
data['property_type'] = data['property_type'].replace(['Dorm', 'Hut', 'Treehouse'], 'Other')

data['property_type'] = data['property_type'].replace(['Camper/RV', 'Timeshare', 'Cabin', 'Hostel', 'In-law', 
    'Boutique hotel', 'Boat', 'Serviced apartment', 'Tent', 'Castle', 'Vacation home', 'Yurt', 
    'Chalet', 'Earth House', 'Tipi', 'Train', 'Cave', 'Parking Space', 
    'Casa particular', 'Lighthouse', 'Island'], 'Other2')

### Dates
data['first_review'] = pd.to_datetime(data['first_review'])
data['last_review'] = pd.to_datetime(data['last_review'])
data['host_since'] = pd.to_datetime(data['host_since'])

data['now'] = datetime.datetime.now()
data['num_days_since_last_review'] = data['now'] - data['last_review']

data['num_days_since_last_review'] = [x.days for x in data['num_days_since_last_review']]

data['host_length'] = [x.days for x in (data['now'] - data['host_since'])]

############# Amenities ############################
import re
f = lambda x : [r for r in re.sub(r'[^,a-z0-9]','',x.lower()).split(',') if len(r) > 1]
amenities = pd.get_dummies(data['amenities'].map(f).apply(pd.Series).stack()).sum(level=0)

data1 = pd.concat([data, amenities], axis=1)

data1a = data1.copy()


# firm mattress = needs to correct error
# doorman = needs to be combined
# accessible = accessibleheightbed, accessibleheighttoilet, 

############# Null values #########################
data1.isnull().sum()


############# Encode ###############################
data1 = pd.get_dummies(data1, columns=['property_type', 'room_type', 'bed_type', 'cancellation_policy',
                                     'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified',
                                     'instant_bookable', 'city'])


############# Drop Vars ###############################
data1 = data1.drop(['amenities', 'first_review', 'host_response_rate', 'last_review', 'review_scores_rating', 
                    'now', 'translationmissingenhostingamenity49', 
                    'translationmissingenhostingamenity50', 'host_since', 'num_days_since_last_review', 
                    'neighbourhood'], axis=1)

data1 = data1.drop(['amenities', 'zipcode', 'now', 'first_review', 'last_review', 
                    'translationmissingenhostingamenity49', 'translationmissingenhostingamenity50'], axis=1)


############# check null ###############################
s = data1.isnull().sum()
s[s>0]

## drop all rows with missing vals
data1 = data1.dropna(axis=0, how='any')

data1.shape #73,036

data1.shape #73,036

############# Modeling #############################
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

data1 = data1.drop(['review_scores_rating', 'host_response_rate'], axis=1)


def run_model(y1, data, model):
    y = data[y1]
    x = data.drop([y1], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    
    # Fit model
    lr = model
    lr.fit(x_train, y_train)
    y_test_pred = lr.predict(x_test)
    return math.sqrt(mean_squared_error(y_test, y_test_pred))


data1.isnull().sum()
run_model('log_price', data1, GradientBoostingRegressor(n_estimators=30, verbose=1, max_depth=5)) 

### Include city; 0.42

################33
neighbor = [c for c in data1 if c.startswith('neighbo')]
d1 = data1.drop(neighbor, axis=1)



###################
# Playing with zip code
os.chdir("C:\\Users\\hanzhu\\Documents\\AirBnB\\Zip Code")

zipcode = pd.read_csv('15zpallagi.csv')

plt.scatter(x=zipcode['A00100'], y=zipcode['A02650'])

# get median income for zipcode
med_income = income.groupby(['zipcode'])['A02650'].apply(np.median)
med_income = pd.DataFrame({'zipcode': med_income.index, 'median_income': med_income.values})
med_income['median_income'] = [str(x) for x in med_income['median_income']]


## Combine income data with data1
data1b = pd.merge(data1, med_income, on='zipcode')






