from pyspark import SparkContext, SparkConf
import sys, math, time
from itertools import combinations
from io import StringIO
import csv

if __name__ == '__main__':
    start_time = time.time()

    # Create a SparkConf and SparkContext
    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)

    # Inputs
    filter_threshold = int(sys.argv[1])
    support_threshold = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    # Inputs
    # filter_threshold = 45
    # support_threshold = 30
    # input_file_path = '/Users/tariqsanda/USC/pycharm/hw2/ta_feng_all_months_merged.csv'
    # output_file_path = './task2_output.txt'

    # Column F (PRODUCT_ID) to numbers (with zero decimal places) in the csv file before reading it using spark.

    # Read the CSV file into an RDD
    rdd = sc.textFile(input_file_path)

    header = rdd.first()  # "TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE","PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"
    data_rdd = rdd.filter(lambda line: line != header)


    def process_line(line):
        # Parse the CSV line
        row = next(csv.reader(StringIO(line)))
        # Create a new CUSTOMER_ID by combining date and original CUSTOMER_ID
        new_customer_id = f"{row[0]}-{int(row[1])}"
        # Return the new key-value pair
        return (new_customer_id, str(int(row[5])))


    parsed_rdd = data_rdd.map(process_line)

    parsed_rdd_to_csv = parsed_rdd.collect()
    output_file_path_csv = './task2_output_inter.csv'
    with open(output_file_path_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['DATE-CUSTOMER_ID', 'PRODUCT_ID'])
        for row in parsed_rdd_to_csv:
            csv_writer.writerow([f'{row[0]}', row[1]])

            # sample_baskets = parsed_rdd.take(5)  # Take the first 5 elements as a sample
    # for basket in sample_baskets:
    #     print(f"Type: {type(basket)}, Content: {basket}")
    # Type: <class 'tuple'>, Content: ('11/1/2000-1104905', 4710199010372)
    # Type: <class 'tuple'>, Content: ('11/1/2000-418683', 4710857472535)

    # parsed_rdd = data_rdd.map(lambda line: next(csv.reader(StringIO(line))))

    # print(parsed_rdd.take(10))

    processing_step1 = parsed_rdd.groupByKey().mapValues(set)

    basketsRDD = processing_step1.map(lambda line: line[1]).filter(lambda x: len(x) > filter_threshold).cache()
    # basketsRDD = processing_step1.map(lambda line: line[1]).filter(lambda x: len(x) > filter_threshold).map(lambda x: {str(x)})
    # .map(lambda x: {item.strip('"') for item in x}).cache())

    # for line in basketsRDD.take(5):
    #     print(line)

    # sample_baskets = basketsRDD.take(5)  # Take the first 5 elements as a sample
    # for basket in sample_baskets:
    #     print(f"Type: {type(basket)}, Content: {basket}")
    # Type: <class 'set'>, Content: {'{19200020480, 4710011432832, 4710777110661, 4710030356102, 50000301829, 2250271000072, 4710320224265, 4712031001098, 4711461660042, 4710508101517, 2250078000398, 4717265888149, 4710011432856, 4711271000472, 4711461668888, 20515485, 4714242832669, 4710028201247, 4710057830050, 4714242832676, 7616100275366, 4710030650408, 4714242832683, 4710144201404, 4710311722817, 20566081, 4711781730630, 4710126392007, 4710095987402, 4710011417419, 20557003, 4710011417426, 4714677353043, 4710043004373, 4711524000600, 4710967990240, 4712500125027, 4710290007011, 4710857000035, 4711587808229, 39101046884, 4711258001256, 4711524000617, 4710861100011, 4710585930604, 4710043100013, 4710311111918, 4901616006126, 4711681000048, 4719598018545, 4710047502066, 4714242862710, 4710095958006, 4710017089016, 4710011432825, 4710857000059, 4711461660028, 4710357304701}'}

    sampled_basketsRDD = basketsRDD.sample(False, 0.1, seed=42)

    count_of_all_baskets = basketsRDD.count()
    # print(f"count_of_all_baskets: {count_of_all_baskets}") # there are 19 items in small1.csv

    # check input
    # for basket in basketsRDD.take(20):
    #     print(f"Basket{basket}")
    num_partitions = basketsRDD.getNumPartitions()
    sampled_basketsRDD = sampled_basketsRDD.repartition(num_partitions)

    # print(f"num_partitions: {num_partitions}")

    local_basket_threshold_val = math.ceil(support_threshold / float(num_partitions))
    # print(f"basket_threshold_val: {basket_threshold_val}")

    single_candidate_items_RDD = basketsRDD.flatMap(lambda x: x).distinct().map(lambda x: {str(x)}).collect()


    # print(single_candidate_items_RDD)
    # sample_baskets = single_candidate_items_RDD.take(5)  # Take the first 5 elements as a sample
    # for basket in sample_baskets:
    #     print(f"Type: {type(basket)}, Content: {basket}")
    # Type: <class 'set'>, Content: {'2250271000072'}
    # Type: <class 'set'>, Content: {'4712031001098'}

    def a_priori_algo(baskets, single_candidates, threshold_val):
        baskets = list(baskets)
        # print(f"baskets: {baskets}")
        # print(f"single_candidates: {single_candidates}")
        # print(f"threshold_val: {threshold_val}")
        frequent_itemsets = []
        k = 2
        while single_candidates:
            frequent_itemsets.append(frequent_items(single_candidates, baskets, threshold_val))
            # print(f"frequent_itemsets: {frequent_itemsets}")
            single_candidates = generate_candidates(frequent_itemsets[-1])
            # print(f"single_candidates: {single_candidates}")
            k += 1

        # print(f"frequent_itemsets: {frequent_itemsets}")
        return frequent_itemsets


    def frequent_items(candidate_itemsets, baskets, support_threshold):
        # print(f"candidate_itemsets: {candidate_itemsets}")
        # print(f"baskets: {baskets}")
        # print(f"support_threshold: {support_threshold}")
        # counts = collections.defaultdict(int)
        counts = {}
        for basket in baskets:
            for candidate in candidate_itemsets:
                # print(f"candidate: {candidate}, basket: {basket}")
                frozenset_candidate = frozenset(candidate)
                # print(f"candidate: {frozenset_candidate}, basket: {basket}")
                if frozenset_candidate.issubset(basket):
                    counts[frozenset(candidate)] = counts.get(frozenset_candidate, 0) + 1

        frequent_itemset = [set(itemset) for itemset, count in counts.items() if count >= support_threshold]
        # print(f"frequent_itemset: {frequent_itemset}")
        return frequent_itemset


    def generate_candidates(prev_candidates):
        candidates = set()

        for a, b in combinations(prev_candidates, 2):
            union_set = set(a).union(b)

            if len(union_set) == len(a) + 1:
                candidates.add(tuple(sorted(union_set)))

        return candidates


    a_priori_phase_1 = basketsRDD.mapPartitions(
        lambda x: a_priori_algo(x, single_candidate_items_RDD, local_basket_threshold_val)) \
        .flatMap(lambda x: x).map(lambda x: (x,) if isinstance(x, str) else tuple(sorted(x))) \
        .map(lambda x: (x,) if isinstance(x, str) and ',' not in x else x).collect()


    # print(f"candidates sampled: {a_priori_phase_1}")
    def check_freq(basket, candidates):
        basket = list(basket)
        count_priori_phase_2 = []
        for candidate in candidates:
            candidate_count = 0
            candidate_set = set(p for p in candidate)
            for user in basket:
                if candidate_set.issubset(user):
                    candidate_count += 1
            count_priori_phase_2.append((candidate, candidate_count))
        # print(f"Counts of candidats in baskets: {count_priori_phase_2}")
        return count_priori_phase_2


    a_priori_phase_2 = basketsRDD.mapPartitions(lambda x: check_freq(x, a_priori_phase_1)).reduceByKey(
        lambda x, y: x + y).filter(lambda x: x[1] >= support_threshold).map(lambda x: x[0]).collect()


    # print(f"All freq items singles, pairs triples etc: {a_priori_phase_2}")

    def format_output(items, w):
        max_length = max(map(len, items))
        formatted_items = [[] for _ in range(max_length)]

        for item in items:
            length = len(item)
            formatted_item = sorted(set(item))  # Sort the item lexicographically
            formatted_items[length - 1].append(formatted_item)

        for length, subsets in enumerate(formatted_items, start=1):
            for subset_index, subset in enumerate(sorted(subsets)):  # Sort subsets lexicographically
                formatted_subset = "(" + ",".join(
                    sorted(map(lambda x: f"'{x}'", subset))) + ")" if length == 1 else str(tuple(subset))
                w.write(formatted_subset)

                # Check if it's the last item in the line
                if subset_index < len(subsets) - 1:
                    w.write(',')
            w.write("\n\n")


    with open(output_file_path, 'w') as text_file:
        text_file.write("Candidates:\n")
        format_output(a_priori_phase_1, text_file)
        text_file.write("Frequent Itemsets:\n")
        format_output(a_priori_phase_2, text_file)

    end_time = time.time()
    print(f"Duration: {end_time - start_time}")