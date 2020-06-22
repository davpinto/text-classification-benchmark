from pandas import read_csv
from sklearn.model_selection import train_test_split
from text_utils import clean_text, clean_text_min
import settings
import logging

## Import dataset
logging.info("Importing dataset...")
data = read_csv("data/ag_news_data.csv")

## Create a single text field
logging.info("Merging text...")
data["text"] = data["title"] + " " + data["description"]

## Split for training and test
logging.info("Spliting data...")
data_tr, data_te, y_tr, y_te = train_test_split(
    data[["text","category"]], 
    data.category, 
    test_size = 0.2, 
    random_state = 0
)

## Clean text and persist on disk
logging.info("Cleaning and formating text...")
data_tr["text"] = data_tr["text"].apply(clean_text_min)
data_te["text"] = data_te["text"].apply(clean_text_min)
data_tr.to_csv("data/data_train_raw.csv")
data_te.to_csv("data/data_test_raw.csv")
data_tr["text"] = data_tr["text"].apply(clean_text)
data_te["text"] = data_te["text"].apply(clean_text)
data_tr.to_csv("data/data_train.csv")
data_te.to_csv("data/data_test.csv")
