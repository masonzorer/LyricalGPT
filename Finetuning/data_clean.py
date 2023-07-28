#  open data file and print first 5 rows
import pandas as pd

# open data file
data = pd.read_csv('./Data/train.csv')

# remove all rows with 'skit' or 'Skit' in the lyrics column
data = data[~data['lyrics'].str.contains('Skit')]
data = data[~data['lyrics'].str.contains('skit')]

# remove any rows with more than 1000 words in the lyrics column
data = data[data['lyrics'].str.split().str.len().lt(1000)]

# add a space after the word 'Lyrics' in the lyrics column
data['lyrics'] = data['lyrics'].str.replace('Lyrics', 'Lyrics ')

# remove all words before the word 'Lyrics' (including 'Lyrics') in the lyrics column
data['lyrics'] = data['lyrics'].str.split('Lyrics').str[1]

# if the first char in the lyrics column is a space, remove it
data['lyrics'] = data['lyrics'].str.lstrip()

# remove the word 'Embed' from the lyrics column 
data['lyrics'] = data['lyrics'].str.split('Embed').str[0]

# remove the number at the end of the lyrics column (either one or two digits)
data['lyrics'] = data['lyrics'].str.rstrip('0123456789')

# print first 5 rows
print(data.head())

# print unique genres
print(data['type'].unique())

# save cleaned data to csv file
#data.to_csv('./Data/train_clean.csv', index=False)

# drop the artist, year, features, id, language_cld3, language_ft columns
data = data.drop(['artist', 'year', 'features', 'id', 'language_cld3', 'language_ft'], axis=1)

# only keep the 2000 most popular songs (using views) from each genre
data = data.groupby('tag').apply(lambda x: x.sort_values('views', ascending=False).head(2000)).reset_index(drop=True)

# split data into a train and validation set with pandas and a 97/3 split
train = data.sample(frac=0.97, random_state=42)
val = data.drop(train.index)

# save train and validation sets to csv files
train.to_csv('./Data/train_clean.csv', index=False)
val.to_csv('./Data/val_clean.csv', index=False)



