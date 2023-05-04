import os
import csv
import json
import pandas as pd
from sklearn.model_selection import train_test_split

data_fields = {
    'userID': 'reviewerID', 
    'itemID': 'asin', 
    'review': 'reviewText', 
    'rating': 'overall'
}
def build_data_profile(data_file, save_folder, data_fields):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder + "/userID")
        os.makedirs(save_folder + "/itemID")
        print("Data folder create")
    else:
        print("Folder already exists.")
    f = open(data_file, "r", encoding="utf-8")
    while True:
        js = f.readline()
        if not js:
            break
        js = json.loads(js)

        userID = js[data_fields['userID']]
        itemID = js[data_fields['itemID']]

        user_review_file = os.path.join(save_folder, "userID", userID + ".txt")
        item_review_file = os.path.join(save_folder, "itemID", itemID + ".txt")
        if data_fields['review'] in js:
            review = js[data_fields['review']]
            review = ' '.join(review.split())
            with open(user_review_file, "a+", encoding = "utf-8") as fu:
                fu.write(review + "\n")
            with open(item_review_file, "a+", encoding = "utf-8") as fi:
                fi.write(review + "\n")
    f.close()
    
def build_rating_profile(data_file, save_folder, data_fields):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Data folder create")
    else:
        print("Folder already exists.")
    f = open(data_file, "r", encoding="utf-8")
    rating_file = os.path.join(save_folder, "rating_full.csv")
    with open(rating_file, 'a+',encoding = "utf-8", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user', 'num_user_review', 'item', 'num_item_review', 'rating'])
    while True:
        js = f.readline()
        if not js:
            break
        js = json.loads(js)

        userID = js[data_fields['userID']]
        itemID = js[data_fields['itemID']]
        rating = js[data_fields['rating']]
        user_review_file = os.path.join(save_folder, "userID", userID + ".txt")
        item_review_file = os.path.join(save_folder, "itemID", itemID + ".txt")

        if os.path.isfile(user_review_file):
            with open(user_review_file, "r", encoding="utf-8") as fu:
                num_user_review = len(fu.readlines())
        else:
            num_user_review = 0
        if os.path.isfile(item_review_file):
            with open(item_review_file, "r", encoding="utf-8") as fi:
                num_item_review = len(fi.readlines())
        else:
            num_item_review = 0
        with open(rating_file, 'a+', encoding = "utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([userID, num_user_review, itemID, num_item_review, rating])
    f.close()

def build_rating_5core(rating_file, save_folder):
    rating_file_5core = os.path.join(save_folder, "rating_5core.csv")
    df = pd.read_csv(rating_file)
    df_5core = df.loc[(df['num_user_review'] >= 5) & (df['num_item_review'] >= 5)]
    df_5core.to_csv(rating_file_5core)

def train_test_val_split(rating_file):
    df = pd.read_csv(rating_file)
    train, valid = train_test_split(df, test_size=0.2, random_state=3)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    train.to_csv(rating_file[:-4] + '_train.csv')
    valid.to_csv(rating_file[:-4] + '_valid.csv')
    test.to_csv(rating_file[:-4] + '_test.csv')