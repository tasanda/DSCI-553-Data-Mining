from pyspark import SparkContext, SparkConf
import sys, collections, math, operator, time, random, csv
from itertools import combinations
from collections import defaultdict

# yelp_train.csv: the training data, which only include the columns: user_id, business_id, and stars.
# user_id,business_id,stars
# vxR_YV0atFxIxfOnF9uHjQ,gTw6PENNGl68ZPUpYWP50A,5.0

time_start = time.time()
conf = SparkConf().setAppName("HW3task1")
sc = SparkContext(conf=conf)

# Inputs
input_file_path = sys.argv[1]
# input_file_path = 'yelp_train.csv'
# output_file_path = 'output_task1.csv'
output_file_path = sys.argv[2]

# Read the CSV file into an RDD
rdd = sc.textFile(input_file_path)

# In this task, we focus on the “0 or 1” ratings rather than the actual ratings/stars from the users.
# Specifically, if a user has rated a business, the user’s contribution in the characteristic matrix is 1.
# If the user hasn’t rated the business, the contribution is 0.

header = rdd.first()
basketsRDD = rdd.filter(lambda line: line != header)

data_rdd = basketsRDD.map(lambda line: line.split(","))

basketsRDD = data_rdd.filter(lambda x: x[2] is not None).map(lambda x: [x[0],x[1]])
# Type: <class 'list'>, Content: ['vxR_YV0atFxIxfOnF9uHjQ', 'gTw6PENNGl68ZPUpYWP50A']
# Type: <class 'list'>, Content: ['o0p-iTC5yTBV5Yab_7es4g', 'iAuOpYDfOTuzQ6OPpEiGwA']

num_rows = basketsRDD.map(lambda x: x[0]).distinct()
unique_list_user_id = num_rows.collect()
# print(unique_list_user_id) # ['o0p-iTC5yTBV5Yab_7es4g', '-qj9ouN0bzMXz1vfEslG-A', '4bQqil4770ey8GfhBgEGuw',...
count_num_rows = num_rows.count() # 11270

# Create a mapping from user_id to a numerical representation
user_id_to_numeric = {user_id: i for i, user_id in enumerate(unique_list_user_id)}

# Now, you can use this mapping to convert user_ids to numerical values
basketsRDD = basketsRDD.map(lambda x: (user_id_to_numeric[x[0]], x[1]))
# Type: <class 'tuple'>, Content: (5626, 'gTw6PENNGl68ZPUpYWP50A')
# Type: <class 'tuple'>, Content: (0, 'iAuOpYDfOTuzQ6OPpEiGwA')
# Type: <class 'tuple'>, Content: (1, '5j7BnXXvlS69uLVHrY9Upw')
# Type: <class 'tuple'>, Content: (5627, 'jUYp798M93Mpcjys_TTgsQ')
# Type: <class 'tuple'>, Content: (5628, '3MntE_HWbNNoyiLGxywjYA')

#convert into shingles
# basketsRDD = basketsRDD.groupByKey().mapValues(set)
basketsRDD = basketsRDD.groupByKey().map(lambda x: (x[0], list(set(x[1]))))
# Type: <class 'tuple'>, Content: (5626, {'ixAh9crILnJ9tM8LhWFhkw', 'owsVSnllxn994EpBR8ZFwQ',
# Type: <class 'tuple'>, Content: (0, {'e_Nf4zAA1KZq80aoy3v8Ng', 'v3AS5LGeV2Si4nOHZ7lgxQ',
# Type: <class 'tuple'>, Content: (5628, {'tCSlpwJQ4CZsUEMZeH2SFg', 'fLdLjrLfwWJ-hh4Uwz2zKA',
# Type: <class 'tuple'>, Content: (2, {'Un_HXZF1JXxmQlYUG3SCTw', 'hrub8NmZuJM-5vO6Rx6P-Q',
# Type: <class 'tuple'>, Content: (5630, {'4_EgrMY-EI-i-xyWixI2qg', 'W5d8iNog90R-qw43m5dGwg',
# sample_baskets = basketsRDD.take(5)  # Take the first 5 elements as a sample
# for basket in sample_baskets:
#     print(f"Type: {type(basket)}, Content: {basket}")

m = 11270 # num_rows = 11270 in training data
p = 99991 # A prime number larger than 'm'
num_hash_funcs = 100  # 128 permutations

def hash_function(user_id):
    hash_values = []
    for i in range(num_hash_funcs):
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        hash_values.append(((a * user_id + b) % p) % m)
    return hash_values

signature_matrix = basketsRDD.map(lambda x: ((x[1], hash_function(x[0])))) \
    .flatMap(lambda x: ((business_id, x[1]) for business_id in x[0])) \
    .groupByKey().map(lambda x: (x[0], [min(col) for col in zip(*x[1])])).collect()

# sample_baskets = signature_matrix.take(5)  # Take the first 5 elements as a sample
# for basket in sample_baskets:
#     print(f"Type: {type(basket)}, Content: {basket}")

# Divide the matrix into b bands with r rows each, where b x r = n (n is the number of hash functions).

# Generate candidate pairs

num_bands = 50
num_rows = 2

candidates = set()
hash_bucket = defaultdict(set)

# Initialize a dictionary to store business IDs and their corresponding user IDs
business_users_dict = data_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

# Iterate through bands and rows to find candidate pairs
for band_index in range(num_bands):
    for signature_id, signature_values in signature_matrix:
        # starting position for the current band
        band_start = band_index * num_rows
        # values within the current band
        band_values = tuple(signature_values[band_start:band_start + num_rows])
        # Hash the values in the band
        hashed_band_value = hash(band_values)
        # Add the signature ID to the corresponding hash bucket
        hash_bucket[hashed_band_value].add(signature_id)

    # Generate candidate pairs from the current band
    for bucket_values in hash_bucket.values():
        # Check if there are at least two values in the bucket
        if len(bucket_values) >= 2:
            # Update the set of candidates with combinations of business IDs
            candidates.update(combinations(sorted(bucket_values), 2))

results = []

for business_id_1, business_id_2 in candidates:
    users1 = business_users_dict.get(business_id_1, set())
    users2 = business_users_dict.get(business_id_2, set())
    if users1 and users2:
        similarity = len(users1 & users2) / len(users1 | users2) if users1 and users2 else 0.0
        if similarity >= 0.5:
            sorted_b1, sorted_b2 = sorted([business_id_1, business_id_2])
            results.append((sorted_b1, sorted_b2, similarity))

# Sort the results lexicographically
results.sort()

# Define the output file header
header = ["business_id_1", " business_id_2", " similarity"]

# Write sorted results to the output file
with open(output_file_path, 'w+', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(header)
    for business_id_1, business_id_2, similarity in results:
        fout.write(f"{business_id_1},{business_id_2},{similarity}\n")

time_end = time.time()
print(f"Duration: {time_end-time_start}")

