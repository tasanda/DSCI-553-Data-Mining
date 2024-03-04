from graphframes import GraphFrame
import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import collect_list, sort_array
import sys
import time

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

start_time = time.time()
filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]
# filter_threshold = 7
# input_file_path = '../resource/asnlib/publicdata/ub_sample_data.csv'
# community_output_file_path = 'output_task1.csv'

sc = SparkContext(appName="task1")
sc.setLogLevel('WARN')
# Create SQLContext from the SparkContext
sqlContext = SQLContext(sc)

rdd = sc.textFile(input_file_path)
header = rdd.first()
rdd = rdd.filter(lambda line: line != header).map(lambda line: line.split(","))

user_business_basket = rdd.groupByKey().map(lambda x: (x[0], list(set(x[1])))).collect()

# Calculate common businesses and filter based on the threshold
edges = set()
vertices = set()
for user1, businesses1 in user_business_basket:
    for user2, businesses2 in user_business_basket:
        if user1 != user2:
            common_businesses = set(businesses1) & set(businesses2)
            if len(common_businesses) >= filter_threshold:
                vertices.add((user1,))
                vertices.add((user2,))
                edges.add((user1, user2))

vertices_df = sqlContext.createDataFrame(list(vertices), ["id"])
edges_df = sqlContext.createDataFrame(list(edges), ["src", "dst"])

graph = GraphFrame(vertices_df, edges_df)

communities_in_graph = graph.labelPropagation(maxIter=5)
# communities_in_graph.show()

communities = communities_in_graph.rdd.map(lambda x: (x[1], x[0])).groupByKey().map(
    lambda x: list(sorted(x[1]))).sortBy(lambda x: (len(x), x)).collect()

with open(community_output_file_path, "w+") as output:
    for community in communities:
        # Convert the community list to a string, remove the first and last characters, and add a new line
        community_label = str(community)[1:-1] + "\n"
        output.write(community_label)

end_time = time.time()

print(f"Duration: {end_time - start_time}")