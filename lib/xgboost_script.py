import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
import pickle

path = ""
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# One-hot-encode categorical variables
train['dataset'] = "train"
test['dataset'] = "test"
data = pd.concat([train,test], axis = 0).reset_index()
categorical = ['property_type','room_type','bed_type','cancellation_policy','city']
data = pd.get_dummies(data, columns = categorical)

#Function to convert amentities string to list
f = lambda x : [r for r in re.sub(r'[^,a-z0-9]','',x.lower()).split(',') if len(r) > 1]
#Amenities list to dummy vars
amenities = pd.get_dummies(data['amenities'].map(f).apply(pd.Series).stack()).sum(level=0)
data = pd.concat([data,amenities],axis=1)


##Some extra features to create from base data
data['host_response_rate'] = data['host_response_rate'].map(lambda x: float(x.split('%')[0])/100 if isinstance(x,str) else 0)
data['instant_bookable'] = data['instant_bookable'].map({'f':0,'t':1})
data['host_has_profile_pic'] = data['host_has_profile_pic'].map({'f':0,'t':1})
data['cleaning_fee'] = data['cleaning_fee'].map({False:0,True:1})


#add rgb data to dataset
rgb = pd.read_csv('./data/withRgb.csv',encoding='iso-8859-1')
data = data.merge(rgb[['id','meanG','meanR','meanB']],left_on='id',right_on='id')

#add median income for census tract to dataset

ct_median_income = pd.read_csv('./data/ct_median_income.csv')
data = data.merge(ct_median_income[['id','ct_median_income']],left_on='id',right_on='id')
data['ct_median_income'] = pd.to_numeric(data['ct_median_income'])

#add zillow data to dataset
zillow = pd.read_csv('./data/Zip_MedianRentalPrice_AllHomes.csv',index_col='RegionName')['2017-12']
zillow.index = [str(zip) for zip in zillow.index]
data['home_prices_zillow'] = data['zipcode'].map(zillow)


#load pickled xgboost model
model = pickle.load(open('./models/xg_model.dat','rb'))

import xgboost as xgb

numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_x = data[data.dataset == "train"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values


test_x = data[data.dataset == "test"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values
    
train_y = data[data.dataset == "train"].log_price.values

dtrain = xgb.DMatrix(train_x, train_y)

print('RMSE:',mean_squared_error(model.predict(dtrain), train_y )**(1/2))
