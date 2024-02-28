from pyspark import SparkContext, SparkConf
import json
import sys

if __name__== '__main__':
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    conf = SparkConf().setAppName("JSONParsingExample")
    sc = SparkContext(conf=conf)

    textRDD = sc.textFile(review_filepath)

    # Count of the total number of reviews
    n_review = textRDD.flatMap(lambda line: [json.loads(line)]).count()

    # Count of the number of reviews in 2018
    n_review_2018 = textRDD.flatMap(lambda line: [json.loads(line)]).filter(lambda review: review.get("date", "").startswith("2018")).count()

    # the number of distinct users who wrote reviews
    n_user = textRDD.flatMap(lambda line: [json.loads(line)]).map(lambda user: user.get("user_id", "")).distinct().count()

    # the top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote - not calculating well
    top10_user = textRDD.flatMap(lambda line: [(json.loads(line)["user_id"], 1)]).reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: -x[1])

    # The number of distinct businesses that have been reviewed
    n_business = textRDD.flatMap(lambda line: [json.loads(line)]).map(lambda user: user.get("business_id", "")).distinct().count()

    # The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business = textRDD.map(lambda line: (json.loads(line)["business_id"], 1)).reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: -x[1])

    task1_results = {
        "n_review": n_review,
        "n_review_2018": n_review_2018,
        "n_user": n_user,
        "top10_user": top10_user,
        "n_business": n_business,
        "top10_business": top10_business
    }

    with open(output_filepath, 'w') as json_file:
        json.dump(task1_results, json_file)
