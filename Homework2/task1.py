from pyspark import SparkContext, SparkConf
import sys, collections, math, operator, time
from itertools import combinations

if __name__ == '__main__':
    start_time = time.time()
    # Create a SparkConf and SparkContext
    conf = SparkConf().setAppName("CSVReadExample")
    sc = SparkContext(conf=conf)

    # Inputs
    case_number = int(sys.argv[1])
    support_threshold = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    # Read the CSV file into an RDD
    rdd = sc.textFile(input_file_path)

    header = rdd.first()
    basketsRDD = rdd.filter(lambda line: line != header)

    # Case for User ID baskets
    if case_number == 1:
        basketsRDD = basketsRDD.map(lambda line: line.split(",")).groupByKey().mapValues(set).map(lambda line: line[1])
    elif case_number == 2:
        basketsRDD = basketsRDD.map(lambda line: tuple(line.split(","))).map(lambda x: [x[1], x[0]]).groupByKey().mapValues(set).map(lambda x: x[1])

    sampled_basketsRDD = basketsRDD.sample(False, 0.1, seed=42)

    count_of_all_baskets = basketsRDD.count()
    # print(f"count_of_all_baskets: {count_of_all_baskets}") # there are 19 items in small1.csv

    num_partitions = basketsRDD.getNumPartitions()
    sampled_basketsRDD = sampled_basketsRDD.repartition(num_partitions)

    # print(f"num_partitions: {num_partitions}")

    local_basket_threshold_val = math.ceil(support_threshold / float(num_partitions))
    # print(f"basket_threshold_val: {basket_threshold_val}")

    # single_candidate_items_RDD = basketsRDD.flatMap(lambda x: x[1]).distinct().collect()

    single_candidate_items_RDD = sampled_basketsRDD.flatMap(lambda x: x).distinct().map(lambda x: {x}).collect()
    # print(f"single_candidate_items_RDD: {single_candidate_items_RDD}")

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

    a_priori_phase_1 = basketsRDD.mapPartitions(lambda x: a_priori_algo(x, single_candidate_items_RDD, local_basket_threshold_val)) \
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


    a_priori_phase_2 = basketsRDD.mapPartitions(lambda x: check_freq(x, a_priori_phase_1)).reduceByKey(lambda x,y: x + y).filter(lambda x: x[1] >= support_threshold).map(lambda x: x[0]).collect()
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
                formatted_subset = "(" + ", ".join(sorted(map(lambda x: f"'{x}'", subset))) + ")" if length == 1 else str(tuple(subset))
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
