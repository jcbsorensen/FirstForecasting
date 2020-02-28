# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# Libraries Settings
pd.set_option("display.max_columns", 100)  # Increase number of columns shown
pd.options.display.float_format = "{:.2f}".format  # formats the number shown in describe()
pd.options.display.max_colwidth = 100  # Increase width of text showns


shops = pd.read_csv("./data/shops.csv")
items = pd.read_csv("./data/items.csv")
itemCategories = pd.read_csv("./data/item_categories.csv")
sales_train = pd.read_csv("./data/sales_train.csv.gz")
sales_test = pd.read_csv("./data/test.csv")


raw_data = pd.read_csv('./data/train_v2.csv')
del raw_data['Unnamed: 0']
subset_data = raw_data.loc[
    raw_data.Store.isin([4]), ['Date', 'Sales', 'Customers', 'Open', 'Month', 'Weekofyear', 'Promo',
                                     'SchoolHoliday', 'Store']]
# Map Day of Week and Set Date to Index
subset_data['Date'] = pd.to_datetime(subset_data['Date'])
subset_data['DoW'] = subset_data['Date'].dt.dayofweek
subset_data.set_index('Date', inplace=True)
# Sort Index to Ascending Dates (low to high)
subset_data.sort_index(inplace=True)

ohe = OneHotEncoder(sparse=False)
ohe.fit(subset_data[['Month']])
labels = ohe.transform(subset_data[['Month']])
labels_df = pd.DataFrame(labels)

subset_data_ohe = pd.get_dummies(subset_data, prefix_sep='_', columns=['Month'], drop_first='True')
