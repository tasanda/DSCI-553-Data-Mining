# Method Description: The recommendation system integrates various data sources to predict user-business ratings, employing XGBoost for regression and a switching hybrid method. Data from JSON files like business, check-in, photo, tip, and user files is processed for features like business attributes, check-in counts, and user interactions. The system utilizes Optuna for XGBoost hyperparameter tuning, enhancing accuracy through feature-engineered data.

# A hybrid approach combines item-based and model-based predictions, favoring item-based predictions when both user and business are in the training set. It optimizes predictions based on their presence in the data. In a switching strategy, it exclusively employs item-based ratings for pairs found in the training data, ensuring better accuracy for well-represented pairs. For absent pairs, the model resorts to model-based ratings, catering to less common or new pairs.

# This strategy offers specialized predictions based on training data availability, enhancing accuracy for known pairs while leveraging a model-based approach for less common ones. It iterates through user-business pairs, switching methods based on their presence in the training set. This adaptive approach refines predictions, relying on explicit data when available and transitioning to a model-based approach for unseen or less represented pairs, potentially improving recommendation accuracy.

# Error Distribution:
# >=0 and <=1: 102311
# >=1 and <=2: 32746
# >=2 and <=3: 6145
# >=3 and <=4: 840
# >=4: 2

# RMSE: 0.9793315284893849

# Execution Time: 377.43919038772583

from pyspark import SparkContext, SparkConf
import sys, collections, math, operator, time, random, csv
from itertools import combinations
from collections import defaultdict
import time, math
import json
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime


def predict_rating(user_id, business_id, bus_user_dict, user_business_dict, bus_avg_dict, bus_user_ratings_dict, w_dict,
                   user_avg_train):
    # Checks if the user has any ratings. If not, returns a default rating of 3.0
    if user_id not in user_business_dict:
        return user_avg_train.get(user_id)
    if business_id not in bus_user_dict:
        return business_avg_train.get(business_id)

    w_i_j_list = item_based_collaborative_filtering(user_id, business_id, bus_user_dict, user_business_dict,
                                                    bus_avg_dict, bus_user_ratings_dict, w_dict)
    # Selects the top 15 similarities based on their weights.
    # if len is greater than 10 pick top 60% or pick all vals with similarity greater than 0
    # Select similarities based on a similarity threshold
    similarity_threshold = 0.0
    w_i_j_list = [(w, r) for w, r in w_i_j_list if w > similarity_threshold]
    if not w_i_j_list:
        # If no similarities meet the threshold, use a fallback method or default
        return 3.0  # You can replace this with a more sophisticated default

    prediction_num = sum(w * r for w, r in w_i_j_list)
    prediction_denum = sum(abs(w) for w, _ in w_i_j_list)

    return prediction_num / prediction_denum if prediction_denum != 0 else 3.0


def item_based_collaborative_filtering(user_id, business_id, bus_user_dict, user_business_dict, bus_avg_dict,
                                       bus_user_ratings_dict, w_dict):
    # Similarity Calculation:
    # Iterates over businesses that the user has rated (bus1).
    # Calculates the similarity (w) between the target business (bus) and each of the businesses the user has rated (bus1).
    # The similarity is computed based on the co-rated users and their ratings.
    # If the similarity has already been calculated for the pair of businesses, it retrieves it from the w_dict.
    # Otherwise, it calculates and stores it for future use.
    w_list = []
    for business_j in user_business_dict[user_id]:
        business_i_j = tuple(sorted((business_j, business_id)))

        corated_users = bus_user_dict[business_id] & bus_user_dict[business_j]
        if len(corated_users) <= 1:
            decay_factor = 0.15
            avg_rating_i = bus_avg_dict.get(business_id, 0.0)
            avg_rating_j = bus_avg_dict.get(business_j, 0.0)
            w_i_j = math.exp(-decay_factor * abs(avg_rating_i - avg_rating_j))
        else:
            w_i_j = pearson_correlation(bus_user_ratings_dict[business_id], bus_user_ratings_dict[business_j],
                                        corated_users)
        w_dict[business_i_j] = w_i_j
        # Constructs a list of tuples containing the similarity weight (w) and the user's rating for the
        # corresponding business (bus_user_r_dict[bus1][user]).
        w_list.append((w_i_j, float(bus_user_ratings_dict[business_j][user_id])))
    return w_list


def pearson_correlation(ratings1, ratings2, co_rated_users):
    r_u_i = [float(ratings1[user]) for user in co_rated_users]
    r_u_j = [float(ratings2[user]) for user in co_rated_users]

    r_i_avg = sum(r_u_i) / len(r_u_i)
    r_j_avg = sum(r_u_j) / len(r_u_j)

    norm_r_u_i = [x - r_i_avg for x in r_u_i]
    norm_r_u_j = [x - r_j_avg for x in r_u_j]

    pearson_numerator = sum([x * y for x, y in zip(norm_r_u_i, norm_r_u_j)])
    pearson_denominator = (math.sqrt(sum([x ** 2 for x in norm_r_u_i]))) * (
        math.sqrt(sum([x ** 2 for x in norm_r_u_j])))

    return pearson_numerator / pearson_denominator if pearson_denominator != 0 else 0


def model_load_and_preprocess_data(sc, folder_path, test_file_name):
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
         float(row['attributes']['RestaurantsPriceRange2']) if row.get('attributes') and 'RestaurantsPriceRange2' in
                                                               row[
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

    # photo.json (business_id, count of the pictures each business has in yelp)
    # Contains photo data including the caption and classification (one of "food", "drink", "menu", "inside" or "outside").
    photo_summary = sc.textFile(folder_path + '/photo.json')
    parsed_photo_summary = photo_summary.map(lambda row: json.loads(row)).map(lambda row: (
        row.get('business_id'),
        1 if row.get('label') else 0
    )).reduceByKey(lambda x, y: x + y)

    parsed_photo_summary_dict = dict(parsed_photo_summary.collect())

    # tip.json (business_id, text)
    # Tips written by a user on a business. Tips are shorter than reviews and tend to convey quick suggestions.
    tip_summary = sc.textFile(folder_path + '/tip.json')
    parsed_tip_summary = tip_summary.map(lambda row: json.loads(row)).map(lambda row: (
        row.get('business_id'),
        1 if row.get('text') else 0
    )).reduceByKey(lambda x, y: x + y)

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
    return X_train, Y_train, X_val, user_bus_prediction


def train_and_predict(X_train, Y_train, X_val, user_bus_prediction):
    param = {
        'learning_rate': 0.1,
        'max_depth': 7,
        'min_child_weight': 7,
        'n_estimators': 350,
        'subsample': 0.9,
        'colsample_bytree': 0.8286797366161494,
        'lambda': 1.6808575731959836,
        'alpha': 2.730040349781095,
        'gamma': 0.07932412737830523
    }

    xgb = XGBRegressor(**param)
    print("fitting the model")
    xgb.fit(X_train, Y_train)
    print("predicting the final result")
    Y_pred = xgb.predict(X_val)

    return Y_pred


start_time = time.time()

conf = SparkConf().setAppName("task2_3")
sc = SparkContext(conf=conf)

# Inputs
folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_path = sys.argv[3]

# Read the CSV file into an RDD
train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
header_train = train_rdd.first()
# row[0] = user_id  row[1] = business_id row[2] = rating
train_rdd = train_rdd.filter(lambda line: line != header_train).map(lambda line: line.split(",")).map(
    lambda row: (row[0], row[1], float(row[2])))

# create an RDD
business_user_train = train_rdd.map(lambda line: (line[1], line[0])).groupByKey().mapValues(set).collectAsMap()

user_business_train = train_rdd.map(lambda line: (line[0], line[1])).groupByKey().mapValues(set).collectAsMap()

business_avg_train = train_rdd.map(lambda line: (line[1], float(line[2]))).groupByKey().mapValues(
    lambda x: sum(x) / len(x)).collectAsMap()

user_avg_train = train_rdd.map(lambda line: (line[0], float(line[2]))).groupByKey().mapValues(
    lambda x: sum(x) / len(x)).collectAsMap()

business_user_ratings_train = train_rdd.map(lambda line: (line[1], (line[0], line[2]))).groupByKey().mapValues(
    lambda x: dict(x)).collectAsMap()

# process the validation data set
val_rdd = sc.textFile(test_file_name)
header_val = val_rdd.first()
validation_rdd = val_rdd.filter(lambda line: line != header_val).map(lambda line: line.split(",")).map(
    lambda line: (line[0], line[1]))


def predict_rating_iterator(line):
    w_dict = {}
    return line[0], line[1], predict_rating(line[0], line[1], business_user_train, user_business_train,
                                            business_avg_train, business_user_ratings_train, w_dict, user_avg_train)


item_baesd_predicted_ratings = validation_rdd.map(predict_rating_iterator).map(
    lambda row: ((row[0], row[1]), row[2])).collectAsMap()

X_train, Y_train, X_val, user_bus_prediction = model_load_and_preprocess_data(sc, folder_path, test_file_name)
Y_pred = train_and_predict(X_train, Y_train, X_val, user_bus_prediction)

# Create a list of (user_id, business_id) tuples
user_bus_pairs = [(row[0], row[1]) for row in user_bus_prediction]

# Create predicted ratings by combining user_bus_pairs and Y_pred
model_based_predicted_ratings = dict(zip(user_bus_pairs, Y_pred))

# Switching approach
combined_predictions = []

for user, business in user_bus_pairs:
    if business in business_user_train and user in user_business_train[user]:
        # Use item-based rating when both user and business are in business_user_ratings_train
        item_based_rating = item_based_predicted_ratings.get((user, business), 0.0)
        combined_prediction = item_based_rating
    else:
        # Use model-based rating when either user or business is not in business_user_ratings_train
        model_based_rating = model_based_predicted_ratings.get((user, business), 0.0)
        combined_prediction = model_based_rating

    combined_predictions.append((user, business, combined_prediction))

result_str = [f"{user},{business},{combined_prediction}" for user, business, combined_prediction in
              combined_predictions]

header = ["user_id", " business_id", " prediction"]
with open(output_file_path, 'w+', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(header)
    for result in result_str:
        user, business, prediction = result.split(',')
        row = [user, business, prediction]
        csv_writer.writerow(row)

# RMSE calcualtion
predicted_ratings = {(user, business): prediction for user, business, prediction in combined_predictions}

actual_ratings_rdd = sc.textFile(test_file_name)
header_actual_ratings = actual_ratings_rdd.first()
actual_ratings = actual_ratings_rdd.filter(lambda line: line != header_actual_ratings).map(
    lambda line: line.split(",")).map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap()


def rmse(predictions, actual_ratings):
    squared_errors = []
    for (user_id, business_id), pred_rating in predictions.items():
        actual_rating = actual_ratings.get((user_id, business_id), None)
        if actual_rating is not None:
            squared_errors.append((pred_rating - actual_rating) ** 2)

    n = len(squared_errors)

    if n == 0:
        return -1  # or any other suitable value to indicate no matching pairs
    mean_squared_error = sum(squared_errors) / n
    rmse_value = math.sqrt(mean_squared_error)
    return rmse_value


rmse_score = rmse(predicted_ratings, actual_ratings)
print(f"RMSE: {rmse_score}")


def error_distribution(predictions, actual_ratings):
    between0_1 = 0
    between1_2 = 0
    between2_3 = 0
    between3_4 = 0
    greater_than_4 = 0

    for (user_id, business_id), pred_rating in predictions.items():
        actual_rating = actual_ratings.get((user_id, business_id), None)

        if actual_rating is not None:
            val = abs(pred_rating - actual_rating)

            if val >= 0 and val < 1:
                between0_1 += 1
            elif val >= 1 and val < 2:
                between1_2 += 1
            elif val >= 2 and val < 3:
                between2_3 += 1
            elif val >= 3 and val < 4:
                between3_4 += 1
            elif val >= 4:
                greater_than_4 += 1
    return {
        "Between 0 and 1": between0_1,
        "Between 1 and 2": between1_2,
        "Between 2 and 3": between2_3,
        "Between 3 and 4": between3_4,
        "Greater than 4": greater_than_4
    }


error_dist = error_distribution(predicted_ratings, actual_ratings)
# Print the results
print("Error Distribution")
for range_name, count in error_dist.items():
    print(f"{range_name}: {count}")

end_time = time.time()
print(f"Duration: {end_time - start_time}")s