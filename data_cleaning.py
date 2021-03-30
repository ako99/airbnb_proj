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

df.to_csv('airbnb_cleaned.csv', index = False)