# -*- coding: utf-8 -*-
"""
Data Cleaning for the NYC Airbnb dataset
@author: Alexander Ngo
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime

df = pd.read_csv("AB_NYC_2019.csv")

def days_between(d1,d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

#Days since last review
df["days_since_last_review"] = df.last_review.apply(lambda x: None if pd.isnull(x) else days_between(x, "2019-07-08"))

#Make note if listing has ever had a review before
df["reviewed_yn"] = df.last_review.apply(lambda x: 0 if pd.isnull(x) else 1)

def contains_any(lst, x):
    for i in lst:
        if i in x.lower():
            return True
    return False
    
#train/subway access
df.name = df.name.astype(str)
df["subway_access"] = df.name.apply(lambda x: 1 if contains_any(["train", "subway"],x) else 0)

#near mall
df["mall"] = df.name.apply(lambda x: 1 if contains_any(["mall", "outlet"],x) else 0)

#near stadium
df["stadium"] = df.name.apply(lambda x: 1 if ('stadium' in x.lower()) else 0)

#near airport
df["airport"] = df.name.apply(lambda x: 1 if contains_any(["jfk", "lga", "laguardia", "airport"],x) else 0)

#length of name
df["name_len"] = df.name.apply(lambda x: len(x))

#drop not so important columns
#keep the name column for the word cloud data_eda.ipynb
irrelevant = ['id','host_id','host_name', 'last_review']
df = df.drop(irrelevant, axis = 1)

#drop listings with a price of 0. They usually indicate that the listing is not available at the moment.
df = df[df['price'] != 0]

#log transformation on price for easier analysis. log+1 to avoid division by zero.
df["price_log"] = np.log(df.price+1)

df.to_csv('airbnb_cleaned.csv', index = False)