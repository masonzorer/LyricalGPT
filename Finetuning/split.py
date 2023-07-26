# load data csv and split into train and val
import pandas as pd

# load data
all_data = pd.read_csv('./Data/df_eng.csv', engine="pyarrow")

# split data into train and val
train_data = all_data.sample(frac=0.95, random_state=0)
val_data = all_data.drop(train_data.index)

# save train and val data
train_data.to_csv('./Data/train.csv', index=False)
val_data.to_csv('./Data/val.csv', index=False)
