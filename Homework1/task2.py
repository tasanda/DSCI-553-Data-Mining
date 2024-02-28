from pyspark import SparkContext, SparkConf
import json
import time
import sys

if __name__== '__main__':
    input_file_path = sys.argv[1]
    output_path = sys.argv[2]
    n_partitions = int(sys.argv[3])

    conf = SparkConf().setAppName("JSONParsingExample")
    sc = SparkContext(conf=conf)

    #input_file_path = '/Users/tariqsanda/USC/pycharm/test_review.json'
    textRDD = sc.textFile(input_file_path)
    #n_partitions = 2

    def partition_function(x):
        return hash(x[0])

    # Default
    default = {}
    timeDefaultStart = time.time()
    default_top10_user = textRDD.flatMap(lambda line: [(json.loads(line)["business_id"], 1)])
    default['n_partition'] = default_top10_user.getNumPartitions()
    default_partition_counts = default_top10_user.mapPartitions(lambda iterator: [sum(1 for _ in iterator)])
    default['n_items'] = default_partition_counts.collect()
    default_top10_user = default_top10_user.reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: -x[1])
    timeDefaultEnd = time.time()
    default['exe_time'] = timeDefaultEnd - timeDefaultStart

    # Customized
    customized = {}
    timeCustomizedStart = time.time()
    customized_top10_user = textRDD.flatMap(lambda line: [(json.loads(line)["business_id"], 1)])
    customized_top10_user = customized_top10_user.partitionBy(n_partitions, partitionFunc=partition_function)
    customized['n_partition'] = customized_top10_user.getNumPartitions()

    customized_partition_counts = customized_top10_user.mapPartitions(lambda iterator: [sum(1 for _ in iterator)])
    customized['n_items'] = customized_partition_counts.collect()

    customized_top10_user = customized_top10_user.reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: -x[1])
    timeCustomizedEnd = time.time()
    customized['exe_time'] = timeCustomizedEnd - timeCustomizedStart

    final_output = {}
    final_output['default'] = default
    final_output['customized'] = customized

    #output_path = '/Users/tariqsanda/USC/pycharm/task2_output.json'

    try:
        with open(output_path, 'w') as json_file:
            json.dump(final_output, json_file)
    except Exception as e:
        print(f"Error writing to JSON file: {str(e)}")
