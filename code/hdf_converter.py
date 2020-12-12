# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:42:08 2020

@author: Kiera
"""
import pandas as pd

# Read CSV
df = pd.read_csv("csv/df_final.csv", index_col=[0])

# Change label to category
df['category'] = df['label']
df.drop(['label'], axis=1, inplace=True)
df.drop(df.columns[0], axis=1)

# Add Additional Index
df["idx"] = pd.RangeIndex(len(df.index))
df["idx"] = range(len(df.index))

df.to_hdf("csv/df_final.h5", "data")