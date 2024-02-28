from pyspark import SparkContext, SparkConf
import json
import time
import sys

if __name__== '__main__':
    review_file_path = sys.argv[1]
    business_file_path = sys.argv[2]
    output_filepath_a = sys.argv[3]
    output_filepath_b = sys.argv[4]

conf = SparkConf().setAppName("JSONParsingExample")
sc = SparkContext(conf=conf)

#review_file_path = '/Users/tariqsanda/USC/pycharm/test_review.json'
#business_file_path = '/Users/tariqsanda/USC/pycharm/business.json'

reviewRDD = sc.textFile(review_file_path)
businessRDD = sc.textFile(business_file_path)

review_data = reviewRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["stars"]))
business_data = businessRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["city"]))

# Combine the review_data and business_data by business_id
review_business_data = review_data.join(business_data)

# Collect the data from the RDD
# data = review_business_data.collect()
# for item in data:
#     business_id, (stars, city) = item
#     print(f"Business ID: {business_id}, Stars: {stars}, City: {city}")

# Business ID: ikCg8xy5JIg_NGPx-MSIDA, Stars: 5.0, City: Calgary

# Create a pair RDD where the key is the city and the value is a tuple (stars, 1)
city_stars_data = review_business_data.map(lambda x: (x[1][1], (x[1][0], 1)))
# data2 = city_stars_data.collect()
# for item in data2:
#     (city, stars) = item
#     print(f"City: {city}, Stars: {stars}")
# City: Pittsburgh, Stars: (1.0, 1)
# City: Chandler, Stars: (5.0, 1)

# Reduce by key to calculate the sum of stars and the count of reviews for each city
city_totals = city_stars_data.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
# data3 = city_totals.collect()
# for item in data3:
#     (city, stars) = item
#     print(f"City: {city}, Stars: {stars}")

# City: Calgary, Stars: (5.0, 1)
# City: Toronto, Stars: (9.0, 3)
# City: Las Vegas, Stars: (37.0, 10)

# Calculate the average stars for each city
city_avg_stars = city_totals.map(lambda x: {"city": x[0], "stars": x[1][0] / x[1][1]})

# Collect the results to the driver
result = city_avg_stars.sortBy(lambda x: (-x["stars"], x["city"])).collect()

# output_path = '/Users/tariqsanda/USC/pycharm/task3_output.txt'

try:
    with open(output_filepath_a, 'w') as text_file:
        text_file.write("city,stars\n")
        for item in result:
            text_file.write(f"{item['city']},{item['stars']}\n")
except Exception as e:
    print(f"Error writing to text file: {str(e)}")


# M1
timeM1Start = time.time()
reviewRDD = sc.textFile(review_file_path)
businessRDD = sc.textFile(business_file_path)
review_data = reviewRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["stars"]))
business_data = businessRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["city"]))
review_business_data = review_data.join(business_data)
city_stars_data = review_business_data.map(lambda x: (x[1][1], (x[1][0], 1)))
city_totals = city_stars_data.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
city_avg_stars = city_totals.map(lambda x: {"city": x[0], "stars": x[1][0] / x[1][1]})
# python sort
city_avg_stars_python = city_avg_stars.collect()
top_10_cities_python = sorted(city_avg_stars_python, key=lambda x: (-x["stars"], x["city"]))[:10]
timeM1End = time.time()
timeM1 = timeM1End - timeM1Start

# M2
timeM2Start = time.time()
reviewRDD = sc.textFile(review_file_path)
businessRDD = sc.textFile(business_file_path)
review_data = reviewRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["stars"]))
business_data = businessRDD.map(lambda line: (json.loads(line)["business_id"], json.loads(line)["city"]))
review_business_data = review_data.join(business_data)
city_stars_data = review_business_data.map(lambda x: (x[1][1], (x[1][0], 1)))
city_totals = city_stars_data.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
city_avg_stars = city_totals.map(lambda x: {"city": x[0], "stars": x[1][0] / x[1][1]})
# python sort
city_avg_stars10 = city_avg_stars.sortBy(lambda x: (-x["stars"], x["city"])).take(10)
timeM2End = time.time()
timeM2 = timeM2End - timeM2Start

task1_results = {
        "m1": timeM1,
        "m2": timeM2,
        "reason": "Initial steps are the same so we can attribute the speed of operation on large data sets to Spark's sortBy transformation and take action improves performance vs Pythons version of sorted which is faster on smaller datasets",
    }

# Define the output file path
# output_path = '/Users/tariqsanda/USC/pycharm/task3_output_2.json'

with open(output_filepath_b, 'w') as json_file:
    json.dump(task1_results, json_file, indent=4)

