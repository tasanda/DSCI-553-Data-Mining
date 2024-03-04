from pyspark import SparkContext, SparkConf
import sys, math, time, csv

def predict_rating(user_id, business_id, bus_user_dict, user_business_dict, bus_avg_dict, bus_user_ratings_dict, w_dict):
    # Checks if the user has any ratings. If not, returns a default rating of 3.0
    if user_id not in user_business_dict:
        return 3.0
    if business_id not in bus_user_dict:
        return 3.0

    w_i_j_list = item_based_collaborative_filtering(user_id, business_id, bus_user_dict, user_business_dict, bus_avg_dict, bus_user_ratings_dict, w_dict)
    # Select similarities based on a similarity threshold
    similarity_threshold = 0.0
    w_i_j_list = [(w, r) for w, r in w_i_j_list if w > similarity_threshold]
    if not w_i_j_list:
        # If no similarities meet the threshold, use a fallback method or default
        return 3.0  # You can replace this with a more sophisticated default

    prediction_num = sum(w * r for w, r in w_i_j_list)
    prediction_denum = sum(abs(w) for w, _ in w_i_j_list)

    return prediction_num / prediction_denum if prediction_denum != 0 else 3.0

def item_based_collaborative_filtering(user_id, business_id, bus_user_dict, user_business_dict, bus_avg_dict, bus_user_ratings_dict, w_dict):
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
        # print(f"user_inter: {corated_users}")
        if len(corated_users) <= 1:
            decay_factor = 0.15
            avg_rating_i = bus_avg_dict.get(business_id, 0.0)
            avg_rating_j = bus_avg_dict.get(business_j, 0.0)
            w_i_j = math.exp(-decay_factor * abs(avg_rating_i - avg_rating_j))
            # print(f"w_i_j len <= 1: {w_i_j}, w_i_j_exp: {w_i_j_exp}")
        else:
            w_i_j = pearson_correlation(bus_user_ratings_dict[business_id], bus_user_ratings_dict[business_j], corated_users)
            # print(f"w_i_j len > 1: {w_i_j}")
        w_dict[business_i_j] = w_i_j
        # Constructs a list of tuples containing the similarity weight (w) and the user's rating for the corresponding business (bus_user_r_dict[bus1][user]).
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
    pearson_denominator = (math.sqrt(sum([x ** 2 for x in norm_r_u_i]))) * (math.sqrt(sum([x ** 2 for x in norm_r_u_j])))

    return pearson_numerator / pearson_denominator if pearson_denominator != 0 else 0

start_time = time.time()

conf = SparkConf().setAppName("task2_1")
sc = SparkContext(conf=conf)

# Inputs
train_file_name = sys.argv[1]
# train_file_name = 'yelp_train.csv'
test_file_name = sys.argv[2]
# test_file_name = 'test_case1.csv'
# output_file_path = 'output_task2_1_test.csv'
output_file_path = sys.argv[3]

folder_path = '../resource/asnlib/publicdata'


# Read the CSV file into an RDD
train_rdd = sc.textFile(train_file_name)

header_train = train_rdd.first()
train_rdd = train_rdd.filter(lambda line: line != header_train).map(lambda line: line.split(",")).map(lambda row: (row[0], row[1], float(row[2])))
# row[0] = user_id  row[1] = business_id
# user_id, business_id, stars
# vxR_YV0atFxIxfOnF9uHjQ,gTw6PENNGl68ZPUpYWP50A,5.0

# create an RDD
business_user_train = train_rdd.map(lambda line: (line[1], line[0])).groupByKey().mapValues(set).collectAsMap()
# format row[0] = business_id and row[1] = user_id

user_business_train = train_rdd.map(lambda line: (line[0], line[1])).groupByKey().mapValues(set).collectAsMap()
# user_id_key = 'wf1GqnKQuvH-V3QN80UOOQ'
# values_for_key = user_business_train.get(user_id_key)
# print(f"Values for key user_business_train {user_id_key}: {values_for_key}")

business_avg_train = train_rdd.map(lambda line: (line[1], float(line[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

business_user_ratings_train = train_rdd.map(lambda line: (line[1], (line[0], line[2]))).groupByKey().mapValues(lambda x: dict(x)).collectAsMap()

# process the validation data set
val_rdd = sc.textFile(test_file_name)
header_val = val_rdd.first()
# user_id,business_id,stars
# wf1GqnKQuvH-V3QN80UOOQ,fThrN4tfupIGetkrz18JOg
validation_rdd = val_rdd.filter(lambda line: line != header_val).map(lambda line: line.split(",")).map(lambda line: (line[0], line[1]))
# business_id, user_id
# wf1GqnKQuvH-V3QN80UOOQ, fThrN4tfupIGetkrz18JOg

# business_id, user_id
def predict_rating_iterator(line):
    w_dict = {}
    return line[0], line[1], predict_rating(line[0], line[1], business_user_train, user_business_train, business_avg_train, business_user_ratings_train, w_dict)

predicted_vals_rdd = validation_rdd.map(predict_rating_iterator)

#predicted_ratings = predicted_vals_rdd.map(lambda row: ((row[0], row[1]), row[2])).collectAsMap()

result_str = predicted_vals_rdd.map(lambda x: f"{x[0]},{x[1]},{x[2]}").collect()

header = ["user_id", " business_id", " prediction"]

with open(output_file_path, 'w+', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(header)
    for result in result_str:
        b1, b2, pred = result.split(',')
        row = [b1, b2, pred]
        csv_writer.writerow(row)

end_time = time.time()
print(f"Duration: {end_time - start_time}")


# for RMSE calcualtion
# actual_ratings_rdd = sc.textFile('yelp_val.csv')
# header_actual_ratings = actual_ratings_rdd.first()
# actual_ratings = actual_ratings_rdd.filter(lambda line: line != header_actual_ratings).map(lambda line: line.split(",")).map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap()

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

# rmse_score = rmse(predicted_ratings, actual_ratings)
# print(f"RMSE: {rmse_score}")