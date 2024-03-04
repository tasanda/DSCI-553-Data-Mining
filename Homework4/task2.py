from pyspark import SparkContext, SparkConf
import time, sys, copy
from collections import defaultdict
from copy import deepcopy

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]


def betweeness_girvan_newman(graph, vertex_list):
    edge_betweenness = defaultdict(float)

    # For each vertex in the list
    for root in vertex_list:
        # Data structures for BFS
        parent = defaultdict(set)
        level = {}
        node_sp_count = defaultdict(float)
        path = []
        queue = [root]
        visited = set()
        visited.add(root)
        level[root] = 0
        # Node Shortest Path Count during the BFS search
        node_sp_count[root] = 1

        # Breadth-First Search (BFS)
        while queue:
            current_node = queue.pop(0)
            path.append(current_node)

            # Explore neighbors
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

                    # Update parent, level, and shortest-path count
                    if neighbor not in parent:
                        parent[neighbor] = set()
                        parent[neighbor].add(current_node)
                    else:
                        parent[neighbor].add(current_node)
                    node_sp_count[neighbor] += node_sp_count[current_node]
                    level[neighbor] = level[current_node] + 1
                elif level[neighbor] == level[current_node] + 1:
                    if neighbor not in parent:
                        parent[neighbor] = set()
                        parent[neighbor].add(current_node)
                    else:
                        parent[neighbor].add(current_node)
                    node_sp_count[neighbor] += node_sp_count[current_node]

        # Initialize vertex and edge weights
        vertex_weights = {}
        edge_weights = defaultdict(float)

        for vertex in path:
            vertex_weights[vertex] = 1

        # Calculate edge weights (edge betweenness)
        for vertex in reversed(path):
            for neighbor in parent[vertex]:
                temp_weight = vertex_weights[vertex] * (node_sp_count[neighbor] / node_sp_count[vertex])
                vertex_weights[neighbor] += temp_weight

                # Update edge weights
                edge_key = tuple(sorted([vertex, neighbor]))
                edge_weights[edge_key] += temp_weight

        # Update edge betweenness
        for key, value in edge_weights.items():
            edge_betweenness[key] += value / 2

    # Sort and return edge betweenness values in descending order
    sorted_edge_betweenness = sorted(edge_betweenness.items(), key=lambda x: (-x[1], x[0]))
    return sorted_edge_betweenness


def community_detection(graph, edge_betweenness):
    threshold = 0.2
    best_modularity = float('-inf')
    best_communities = []
    vertices_list = list(graph.keys())
    subgraph = deepcopy(graph)

    def find_community(root, subgraph, visited):
        community = [root]
        visited.add(root)
        queue = [root]
        while queue:
            current_node = queue.pop(0)
            neighbors = subgraph[current_node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    community.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        return community

    while edge_betweenness:

        current_communities = []
        visited = set()

        for vertex in vertices_list:
            if vertex not in visited:
                community = find_community(vertex, subgraph, visited)
                current_communities.append(sorted(community))

        current_communities = sorted(current_communities, key=lambda x: (len(x), x))

        current_modularity = calculate_modularity(graph, current_communities)
        # print(f"Current Modularity: {current_modularity}, Best Modularity: {best_modularity}")

        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = deepcopy(current_communities)
        # elif current_modularity - best_modularity <= threshold:
        #     break

        # Remove the edge with the highest betweenness
        highest_betweenness = edge_betweenness[0][1]
        edge, _ = edge_betweenness.pop(0)
        subgraph[edge[0]].remove(edge[1])
        subgraph[edge[1]].remove(edge[0])

        # Recalculate edge betweenness for the updated subgraph
        edge_betweenness = betweeness_girvan_newman(subgraph, vertices_list)

    return best_communities


def calculate_modularity(graph, communities):
    num_edges = 0  # Initialize the count of edges
    m = 0  # Initialize the total number of edges

    # Calculate the total number of edges and count the edges
    for neighbors in graph.values():
        num_edges += len(neighbors)  # Count the edges connected to each node
    m = num_edges // 2  # Total number of edges (each edge counted twice)

    if m == 0:
        return 0  # Return 0 if there are no edges in the graph

    modularity = 0.0

    for community in communities:
        for i in community:
            neighbors_i = graph.get(i, set())  # Get neighbors of node i
            for j in community:
                neighbors_j = graph.get(j, set())  # Get neighbors of node j
                A_ij = 1 if j in neighbors_i else 0  # Check if there's an edge between i and j
                K_i = len(neighbors_i)
                K_j = len(neighbors_j)
                modularity += (A_ij - K_i * K_j / (2 * m))

    modularity /= (2 * m)
    return modularity


sc = SparkContext(appName="task2")
sc.setLogLevel('WARN')

start_time = time.time()

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
                vertices.add((user1))
                vertices.add((user2))
                edges.add((user1, user2))

# Create a graph using RDD
# Step 1: Create the graph structure
graph = {}  # Initialize an empty graph as a dictionary
# Parse your data and populate the graph
for user1, user2 in edges:
    if user1 not in graph:
        graph[user1] = set()
    if user2 not in graph:
        graph[user2] = set()
    graph[user1].add(user2)
    graph[user2].add(user1)

edge_betweenness = betweeness_girvan_newman(graph, vertices)

# print to file
with open(betweenness_output_file_path, "w") as f:
    for user, value in edge_betweenness:
        user = str(user)
        temp_str = user + "," + str(round(value, 5)) + "\n"
        f.write(temp_str)

best_communities = community_detection(graph, edge_betweenness)

# Convert the result to an RDD and apply transformations
# communities_rdd = sc.parallelize(best_communities)
# sorted_communities = communities_rdd.map(lambda x: (len(x), x)).sortBy(lambda x: (x[0], x[1])).collect()

# # print to file
with open(community_output_file_path, 'w+') as fout:
    for line in best_communities:
        fout.write(str(line).strip('[]') + '\n')

end_time = time.time()
print(f"Duration: {end_time - start_time}")
