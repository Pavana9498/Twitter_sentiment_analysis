import pandas as pd
import numpy as np
import re
import sys

def remove_tags(row):
    row = str(row)
    cleanedRow = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanedRow, ' ', row)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',' ',cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+','',cleantext)
    cleantext = re.sub('\s+',' ',cleantext)
    cleantext = cleantext[1:].strip()
    return cleantext

def preprocess_text(row):
	row = str(row).encode('ascii', 'ignore').strip()
	row = remove_tags(row)

	return row

train_data_path = r"data/training-Obama-Romney-tweets1.xlsx"

obama_df = pd.read_excel(train_data_path, 'Obama')
romney_df = pd.read_excel(train_data_path, 'Romney')

obama_df['tweet'] = obama_df['Anootated tweet'].apply(preprocess_text)
romney_df['tweet'] = romney_df['Anootated tweet'].apply(preprocess_text)

obama_df = obama_df.drop(['Anootated tweet','Unnamed: 5','Unnamed: 6','Unnamed: 0'], axis = 1)
obama_df = obama_df.rename(columns = {'Unnamed: 4':'label'})
obama_df = obama_df[obama_df.label!='Class']
obama_df = obama_df[['date','time','tweet','label']]

romney_df = romney_df.drop(['Anootated tweet','Unnamed: 5','Unnamed: 6','Unnamed: 0'], axis = 1)
romney_df = romney_df.rename(columns = {'Unnamed: 4':'label'})
romney_df = romney_df[romney_df.label!='Class']
romney_df = romney_df[['date','time','tweet','label']]



print("Writing cleaned training output")

obama_df.to_csv('data/obama_csv.csv', sep='\t')
romney_df.to_csv('data/romney_csv.csv', sep='\t')

test_data_path = r"data/testing-Obama-Romney-tweets.xlsx"

obama_df = pd.read_excel(test_data_path, 'Obama')
romney_df = pd.read_excel(test_data_path, 'Romney')

obama_df['tweet'] = obama_df['tweet'].apply(preprocess_text)
romney_df['tweet'] = romney_df['tweet'].apply(preprocess_text)

print("Writing cleaned testing output")

obama_df.to_csv('data/obama_test_csv.csv', sep='\t')
romney_df.to_csv('data/romney_test_csv.csv', sep='\t')


