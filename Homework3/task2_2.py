from pyspark import SparkContext, SparkConf
import sys, collections, math, operator, time, random, csv
import json
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime

conf = SparkConf().setAppName("HW3task2_2")
sc = SparkContext(conf=conf)

# Inputs
folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_path = sys.argv[3]

start_time = time.time()

# Read the training file into an RDD
train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
header_train = train_rdd.first()
train_rdd = train_rdd.filter(lambda line: line != header_train).map(lambda line: line.split(","))
# user_id, business_id, stars
# vxR_YV0atFxIxfOnF9uHjQ,gTw6PENNGl68ZPUpYWP50A,5.0

# read each of the json file for features: business.json

# business.json (business_id, stars, review_count, RestaurantsDelivery, RestaurantsPriceRange2, RestaurantsReservations)
business_summary = sc.textFile(folder_path + '/business.json')
parsed_business_summary = business_summary.map(lambda row: json.loads(row)).map(lambda row: (
    row.get('business_id', 'N/A'),
    (float(row.get('stars', 0.0)),
     float(row.get('review_count', 0)),
     float(row.get('is_open', 0)),  # Default to 0 if 'is_open' is missing
     1 if row.get('attributes') and row['attributes'].get('Alcohol') != "none" else 0,
     1 if row.get('attributes') and row['attributes'].get('BusinessAcceptsCreditCards') == "True" else 0,
     1 if row.get('attributes') and row['attributes'].get('NoiseLevel') != "loud" else 0,
     1 if row.get('attributes') and row['attributes'].get('RestaurantsDelivery') == "True" else 0,
     float(row['attributes']['RestaurantsPriceRange2']) if row.get('attributes') and 'RestaurantsPriceRange2' in row[
         'attributes'] else 0,
     1 if row.get('attributes') and row['attributes'].get('RestaurantsReservations') == "True" else 0,
     )))
# for item in parsed_business_summary.take(5):
#     print(f"Type: {type(item)}, Content: {item}")

business_summary_dict = dict(parsed_business_summary.collect())

# checkin.json (business_id, count of checkins)
checkin_summary = sc.textFile(folder_path + '/checkin.json')
parsed_checkin_summary = checkin_summary.map(lambda row: json.loads(row)).map(lambda row: (
    row.get('business_id'),
    float(len(row['time']))
))
checkin_summary_dict = dict(parsed_checkin_summary.collect())
# {"time":{"Mon-13":1,"Thu-13":1,"Sat-16":1,"Wed-17":1,"Sun-19":1,"Thu-20":1,"Sat-21":1},
# "business_id":"kREVIrSBbtqBhIYkTccQUg"}
# for item in parsed_checkin_summary.collect():
#     print(f"Type: {type(item)}, Content: {item}")

# photo.json (business_id, count of the pictures each business has in yelp)
# Contains photo data including the caption and classification (one of "food", "drink", "menu", "inside" or "outside").
photo_summary = sc.textFile(folder_path + '/photo.json')
parsed_photo_summary = photo_summary.map(lambda row: json.loads(row)).map(lambda row: (
    row.get('business_id'),
    1 if row.get('label') else 0
)).reduceByKey(lambda x, y: x + y)

parsed_photo_summary_dict = dict(parsed_photo_summary.collect())

# for item in parsed_photo_summary.collect():
#     print(f"row: {item}")

# {"photo_id": "H5QslOOUmwVpNSimi6stVA", "business_id": "6xGlz2tG5fjSNOsN2kU5Bw",
# "caption": "Making wontons.", "label": "food"}
# {"photo_id": "PP8L-oekYdC7b6XIQb_r1Q", "business_id": "6xGlz2tG5fjSNOsN2kU5Bw",
# "caption": "", "label": "food"}

# tip.json (business_id, text)
# Tips written by a user on a business. Tips are shorter than reviews and tend to convey quick suggestions.
tip_summary = sc.textFile(folder_path + '/tip.json')
parsed_tip_summary = tip_summary.map(lambda row: json.loads(row)).map(lambda row: (
    row.get('business_id'),
    1 if row.get('text') else 0
)).reduceByKey(lambda x, y: x + y)
# {"text":"Pool table. No charge!","date":"2016-03-17","likes":0,"business_id":"IPpz3dROk6PBhiYM-DEISw","user_id":"BLzD9wKv7fhAHWKnUCzd1w"}
# for item in parsed_tip_summary.collect():
#     print(f"row: {item}")
parsed_tip_summary_dict = dict(parsed_tip_summary.collect())

# user.json (user_id, review_count, yelping_since, friends, average_stars, "elite":"2012, 2013")
# User data including the user's friend mapping and all the metadata associated with the user.
user_summary = sc.textFile(folder_path + '/user.json')
parsed_user_summary = user_summary.map(lambda row: json.loads(row)).map(lambda row: (
    row.get('user_id'),
    (float(row.get('review_count')),
     float((datetime.now() - datetime.strptime(row.get('yelping_since'), "%Y-%m-%d")).days),  # days as a member
     float(len(row.get('friends'))),  # count of friends
     float(row.get('useful') + row.get('funny') + row.get('cool') + row.get('fans')),
     float(len(row.get('elite'))),
     float(row.get('average_stars')),
     float(row.get('compliment_hot') + row.get('compliment_more') + row.get('compliment_profile') + row.get(
         'compliment_cute') + row.get('compliment_list') + row.get('compliment_note') + row.get(
         'compliment_plain') + row.get('compliment_cool') + row.get('compliment_funny') + row.get(
         'compliment_writer') + row.get('compliment_photos'))
     )))

parsed_user_summary_dict = dict(parsed_user_summary.collect())

# for item in parsed_user_summary.take(5):
#     print(f"Type: {type(item)}, Content: {item}")

print("Created all Dicts")
X_train = []
Y_train = []

for user, bus, rating in train_rdd.collect():
    Y_train.append(rating)
    # Get business features
    business_features = business_summary_dict.get(bus, (0, 0, 0, 0, 0, 0, 0, 0, 0))
    # Get checkin features
    checkin_features = (checkin_summary_dict.get(bus, 0),)
    # Get photo features
    photo_features = (parsed_photo_summary_dict.get(bus, 0),)
    # Get tip features
    tip_features = (parsed_tip_summary_dict.get(bus, 0),)
    # Get user features
    user_features = parsed_user_summary_dict.get(user, (0, 0, 0, 0, 0, 0, 0))
    # Combine features
    combined_features = business_features + checkin_features + photo_features + tip_features + user_features
    X_train.append(combined_features)

print("Created X_Train and Y_train Successfully ")
X_train = np.array(X_train, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')

# process the validation data set
val_rdd = sc.textFile(test_file_name)
header_val = val_rdd.first()
# user_id,business_id,stars
# wf1GqnKQuvH-V3QN80UOOQ,fThrN4tfupIGetkrz18JOg
validation_rdd = val_rdd.filter(lambda line: line != header_val).map(lambda line: line.split(",")).map(
    lambda x: [x[0], x[1]])

X_val = []
user_bus_prediction = []

for user, bus in validation_rdd.collect():
    user_bus_prediction.append((user, bus))
    # Get business features
    business_features = business_summary_dict.get(bus, (0, 0, 0, 0, 0, 0, 0, 0, 0))
    # Get checkin features
    checkin_features = (checkin_summary_dict.get(bus, 0),)
    # Get photo features
    photo_features = (parsed_photo_summary_dict.get(bus, 0),)
    # Get tip features
    tip_features = (parsed_tip_summary_dict.get(bus, 0),)
    # Get user features
    user_features = parsed_user_summary_dict.get(user, (0, 0, 0, 0, 0, 0, 0))
    # Combine features
    combined_features = business_features + checkin_features + photo_features + tip_features + user_features
    X_val.append(combined_features)

X_val = np.array(X_val, dtype='float32')

# //////////////////////
# hyper-parameter tuning using google collab
# param_grid = {
#     'learning_rate': [0.1],
#     'max_depth': [5],
#     'min_child_weight': [2],
#     'n_estimators': [350],
#     'subsample': [0.9],
#     'colsample_bytree': [0.7],
#     'nthread': [3],
#     'scale_pos_weight': [1],
#     'seed': [27],
#     'lambda': [1.0],  # Adjust the range as needed
#     'alpha': [9.0],   # Adjust the range as needed
#     'gamma': [0],
# }
# # Create a GridSearchCV object
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# # Perform the grid search
# grid_search.fit(X_train, Y_train)
# # Print the best hyperparameters found
# print("Best hyperparameters:", grid_search.best_params_)
# //////////////////////

param = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 2,
    'n_estimators': 350,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'nthread': 3,
    'scale_pos_weight': 1,
    'seed': 27,
    'lambda': 1.0,
    'alpha': 9.0,
    'gamma': 0,
}

xgb = XGBRegressor(**param)
print("fitting the model")
xgb.fit(X_train, Y_train)
print("predicting the final result")
Y_pred = xgb.predict(X_val)

# Create a list of (user_id, business_id) tuples
user_bus_pairs = [(user, bus) for user, bus in user_bus_prediction]

# Create predicted ratings by combining user_bus_pairs and Y_pred
predicted_ratings = {pair: rating for pair, rating in zip(user_bus_pairs, Y_pred)}

# Generate the result string
result_str = "user_id, business_id, prediction\n"
for (user, bus), rating in zip(user_bus_pairs, Y_pred):
    result_str += f"{user},{bus},{rating}\n"

with open(output_file_path, "w") as f:
    f.writelines(result_str)

end_time = time.time()
print(f"Duration: {end_time - start_time}")


# for RMSE calculation
actual_ratings_rdd = sc.textFile(folder_path + '/yelp_val.csv')
header_actual_ratings = actual_ratings_rdd.first()
actual_ratings = actual_ratings_rdd.filter(lambda line: line != header_actual_ratings).map(lambda line: line.split(",")).map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap()

def rmse(predictions, actual_ratings):
    squared_errors = []
    for (user_id, business_id), pred_rating in predictions.items():
        actual_rating = actual_ratings.get((user_id, business_id), None)
        if actual_rating is not None:
            squared_errors.append((pred_rating - actual_rating) ** 2)

    n = len(squared_errors)

    if n == 0:
        return -1
    mean_squared_error = sum(squared_errors) / n
    rmse_value = math.sqrt(mean_squared_error)
    return rmse_value


rmse_score = rmse(predicted_ratings, actual_ratings)
print(f"RMSE: {rmse_score}")