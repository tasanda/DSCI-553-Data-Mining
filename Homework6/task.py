import numpy as np
from sklearn.cluster import KMeans
import time, sys
from pyspark import SparkContext, SparkConf

# INPUTS
input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]
#input_file = '../resource/asnlib/publicdata/hw6_clustering.txt'
#n_cluster = 10
#output_file = 'task_output.txt'

# variables
rs_outliers = set()
ds_cluster_stats = {}
ds_centroid = {}
ds_mahalanobis_distance = {}
cs_outliers = {}
ds_cluster_points = {}
cs_cluster_points = {}
cs_cluster_stats = {}
cs_centroid = {}
cs_mahalanobis_distance = {}
num_clusters = n_cluster * 5

# Step 1. Load 20% of the data randomly.
data = np.genfromtxt(input_file, delimiter=',')
#np.random.seed(42)
np.random.shuffle(data)
np_data_set = np.array_split(data, 5)
np_data_set_1 = np_data_set[0]

# Step 2. Run K-Means with a large K on the data in memory using Euclidean distance.
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(np_data_set[0][:, 2:])

# Get cluster predictions and centroids
cluster_labels = kmeans.labels_

# Step 3. Identify clusters with only one point as RS (outliers).
rs_clusters = {}
for point_id, cluster_label in enumerate(cluster_labels):
    if cluster_label not in rs_clusters.keys():
        rs_clusters[cluster_label] = []
    rs_clusters[cluster_label].append(point_id)
for point_id in rs_clusters.values():
    if len(point_id) == 1:
        rs_outliers.add(point_id[0])

#print(f"RS Outliers: {rs_outliers}")

# Step 4. Run K-Means again to cluster the rest of the data points excluding outlier clusters.
data_without_rs_outliers = np.delete(np_data_set_1, list(rs_outliers), axis=0)
kmeans_remaining = KMeans(n_clusters=n_cluster)
kmeans_remaining.fit(data_without_rs_outliers[:, 2:])
# Get cluster predictions and centroids for the remaining data
remaining_cluster_labels = kmeans_remaining.labels_
new_clusters = {}
for point_id, cluster_label in enumerate(remaining_cluster_labels):
    if cluster_label not in new_clusters.keys():
        new_clusters[cluster_label] = []
    new_clusters[cluster_label].append(point_id)

# Step 5. Collect statistics for the new clusters.
for point_id, value in new_clusters.items():
    # Calculate summary statistics for the cluster
    num_points = len(data_without_rs_outliers[value, 2:])
    sum_coords = np.sum(data_without_rs_outliers[value, 2:], axis=0)
    sum_squares_coords = np.sum(np.square(data_without_rs_outliers[value, 2:]), axis=0)

    # Store cluster statistics in the dictionary
    ds_cluster_stats[point_id] = {
        'Number of Points': num_points,
        'Vector SUM': sum_coords,
        'Vector SUMSQ': sum_squares_coords
    }
    ds_cluster_points[point_id] = list(map(int, data_without_rs_outliers[value, 0]))

    new_centroid = sum_coords / num_points
    ds_centroid[point_id] = new_centroid
    mahalanobis_distance = np.sqrt(np.subtract(sum_squares_coords / num_points, np.square(new_centroid)))
    ds_mahalanobis_distance[point_id] = mahalanobis_distance


# skip step 6 for now
# ////////////
# Calculating the number of DS clusters
number_discard_points = sum(stats['Number of Points'] for stats in ds_cluster_stats.values())
# Calculating the number of CS outliers and the total number of points in CS outliers
number_clusters_compression_set = len(cs_outliers)
number_compression_points = sum(stats['Number of Points'] for stats in cs_outliers.values())
# Calculating the number of RS clusters and the total number of points in RS outliers
number_points_retained_set = len(rs_outliers)

# Writing intermediate results to the output file
with open(output_file, "w") as f:
    f.write('The intermediate results:\n')
    result_str = f'Round 1: {number_discard_points},{number_clusters_compression_set},{number_compression_points},{number_points_retained_set}\n'
    f.write(result_str)

# Step 7. Load another 20% of the data randomly.
D = 2 * np.sqrt(np_data_set_1.shape[1] - 2)
for round in range(1,5):
    np_data_set_round = np_data_set[round]

    for point_id, value in enumerate(np_data_set_round):
        min_distance_ds = float('inf')
        assigned_cluster_ds = -1
        min_distance_cs = float('inf')
        assigned_cluster_cs = -1

        for cluster_label, centroid in ds_centroid.items():
            mahalanobis_dis = np.sqrt(np.sum(np.square(np.divide(np.subtract(value[2:], centroid), ds_mahalanobis_distance[cluster_label])), axis=0))

            if mahalanobis_dis < min_distance_ds:
                min_distance_ds = mahalanobis_dis
                assigned_cluster_ds = cluster_label

        for cluster_label, summary in cs_cluster_stats.items():
            mahalanobis_dis = np.sqrt(np.sum(np.square(np.divide(np.subtract(value[2:], cs_centroid[cluster_label]), cs_mahalanobis_distance[cluster_label])), axis=0))

            if mahalanobis_dis < min_distance_cs:
                min_distance_cs = mahalanobis_dis
                assigned_cluster_cs = cluster_label

        if min_distance_ds < D and assigned_cluster_ds != -1:
            # Assign the point to the nearest DS cluster
            num_points = ds_cluster_stats[assigned_cluster_ds]['Number of Points'] + 1
            sum_coords = np.add(ds_cluster_stats[assigned_cluster_ds]['Vector SUM'], value[2:])
            sum_squares_coords = np.add(ds_cluster_stats[assigned_cluster_ds]['Vector SUMSQ'], np.square(value[2:]))

            ds_cluster_stats[assigned_cluster_ds]['Number of Points'] = num_points
            ds_cluster_stats[assigned_cluster_ds]['Vector SUM'] = sum_coords
            ds_cluster_stats[assigned_cluster_ds]['Vector SUMSQ'] = sum_squares_coords

            new_centroid = sum_coords / num_points
            ds_centroid[assigned_cluster_ds] = new_centroid

            mahalanobis_distance = np.sqrt(np.subtract(sum_squares_coords / num_points, np.square(new_centroid)))
            ds_mahalanobis_distance[assigned_cluster_ds] = mahalanobis_distance

            ds_cluster_points[assigned_cluster_ds].append(int(value[0]))

        elif min_distance_cs < D and assigned_cluster_cs != -1:
            # Assign the point to the nearest CS cluster
            num_points = cs_cluster_stats[assigned_cluster_cs]['Number of Points'] + 1
            sum_coords = np.add(cs_cluster_stats[assigned_cluster_cs]['Vector SUM'], value[2:])
            sum_squares_coords = np.add(cs_cluster_stats[assigned_cluster_cs]['Vector SUMSQ'], np.square(value[2:]))

            cs_cluster_stats[assigned_cluster_cs]['Number of Points'] = num_points
            cs_cluster_stats[assigned_cluster_cs]['Vector SUM'] = sum_coords
            cs_cluster_stats[assigned_cluster_cs]['Vector SUMSQ'] = sum_squares_coords

            new_centroid = sum_coords / num_points
            cs_centroid[assigned_cluster_cs] = new_centroid

            mahalanobis_distance = np.sqrt(np.subtract(sum_squares_coords / num_points, np.square(new_centroid)))
            cs_mahalanobis_distance[assigned_cluster_cs] = mahalanobis_distance

            cs_cluster_points[assigned_cluster_cs].append(int(value[0]))
        else:
            # print(f"point_id: {point_id}")
            rs_outliers.add(point_id)
        # print(f"RS Outliers: {rs_outliers}")
        # step 11
        # Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
        # Filter indices corresponding to rs_outliers
        # change 1 to round variable
    data_with_only_rs_outliers = np_data_set[round][list(rs_outliers), :]

    # Run K-means on the points identified as outliers (RS)
    if len(rs_outliers) >= num_clusters:
        new_cs_cluster = {}
        kmeans_outliers = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans_outliers.fit(data_with_only_rs_outliers[:, 2:])

        # Get cluster predictions and centroids for the outliers to generate CS and DS
        outlier_cluster_labels = kmeans_outliers.labels_
        for point_id, cluster_id in enumerate(outlier_cluster_labels):
            if cluster_id not in new_cs_cluster.keys():
                new_cs_cluster[cluster_id] = []
                new_cs_cluster[cluster_id].append(point_id)
        rs_outliers = set()
        for id in new_cs_cluster.values():
            if len(id) == 1:
                rs_outliers.add(id[0])

        for cluster_id, point_id in new_cs_cluster.items():
            if len(point_id) > 1:
                num_points = len(point_id)
                sum_coords = np.sum(data_with_only_rs_outliers[point_id, 2:], axis=0)
                sum_squares_coords = np.sum(np.square(data_with_only_rs_outliers[point_id, 2:]), axis=0)
                cs_cluster_stats[assigned_cluster_cs]['Number of Points'] = num_points
                cs_cluster_stats[assigned_cluster_cs]['Vector SUM'] = sum_coords
                cs_cluster_stats[assigned_cluster_cs]['Vector SUMSQ'] = sum_squares_coords

                cs_cluster_points[cluster_id] = list(map(int, data_with_only_rs_outliers[point_id, 0]))

                new_centroid = sum_coords / num_points
                cs_centroid[assigned_cluster_cs] = new_centroid

                mahalanobis_distance = np.sqrt(np.subtract(sum_squares_coords / num_points, np.square(new_centroid)))
                cs_mahalanobis_distance[assigned_cluster_cs] = mahalanobis_distance

    # Step12.Merge CS clusters that have a Mahalanobis Distance < 2 𝑑.
    def merge_clusters(cluster_stats, centroid, mahalanobis_distance, point_dict):
        merged_clusters = {}
        for cluster1 in cluster_stats.keys():
            for cluster2 in cluster_stats.keys():
                if cluster1 != cluster2:
                    mahalanobis_dis1 = calculate_mahalanobis(centroid[cluster1], centroid[cluster2],
                                                             mahalanobis_distance[cluster2])
                    mahalanobis_dis2 = calculate_mahalanobis(centroid[cluster2], centroid[cluster1],
                                                             mahalanobis_distance[cluster1])
                    mahalanobis_dis = min(mahalanobis_dis1, mahalanobis_dis2)

                    if mahalanobis_dis < D:
                        D = mahalanobis_dis
                        merged_clusters[cluster1] = cluster2

        for cluster1, cluster2 in merged_clusters.items():
            if cluster1 in cluster_stats and cluster2 in cluster_stats:
                if cluster1 != cluster2:
                    # Merge clusters
                    merge_two_clusters(cluster_stats, centroid, mahalanobis_distance, point_dict, cluster1, cluster2)


    def calculate_mahalanobis(centroid1, centroid2, mahalanobis_distance):
        return np.sqrt(np.sum(np.square(
            np.divide(np.subtract(centroid1, centroid2), mahalanobis_distance,
                      out=np.zeros_like(np.subtract(centroid1, centroid2)),
                      where=mahalanobis_distance != 0)), axis=0))


    def merge_two_clusters(cluster_stats, centroid, mahalanobis_distance, point_dict, cluster1, cluster2):
        num_points = cluster_stats[cluster1]['Number of Points'] + cluster_stats[cluster2]['Number of Points']
        sum_coords = np.add(cluster_stats[cluster1]['Vector SUM'], cluster_stats[cluster2]['Vector SUM'])
        sum_squares_coords = np.add(cluster_stats[cluster1]['Vector SUMSQ'], cluster_stats[cluster2]['Vector SUMSQ'])

        cluster_stats[cluster2]['Number of Points'] = num_points
        cluster_stats[cluster2]['Vector SUM'] = sum_coords
        cluster_stats[cluster2]['Vector SUMSQ'] = sum_squares_coords

        new_centroid = sum_coords / num_points
        centroid[cluster2] = new_centroid

        mahalanobis_distance = np.sqrt(np.subtract(sum_squares_coords / num_points, np.square(new_centroid)))
        mahalanobis_distance[cluster2] = mahalanobis_distance

        point_dict[cluster2].extend(point_dict[cluster1])

        cluster_stats.pop(cluster1, None)
        mahalanobis_distance.pop(cluster1, None)
        centroid.pop(cluster1, None)
        point_dict.pop(cluster1)


    # Merge CS clusters
    merge_clusters(cs_cluster_stats, cs_centroid, cs_mahalanobis_distance, cs_cluster_points)

    # Merge CS clusters with DS clusters
    if round == 5:
        merge_clusters(cs_cluster_stats, cs_centroid, cs_mahalanobis_distance, cs_cluster_points)

    # Calculating the number of DS clusters
    number_discard_points = number_clusters_compression_set = number_compression_points = number_points_retained_set = 0

    number_discard_points = sum(stats['Number of Points'] for stats in ds_cluster_stats.values())

    # Calculating the number of CS outliers and the total number of points in CS outliers
    number_clusters_compression_set = len(cs_outliers)
    number_compression_points = sum(stats['Number of Points'] for stats in cs_outliers.values())

    # Calculating the number of RS clusters and the total number of points in RS outliers
    number_points_retained_set = len(rs_outliers)

    # Writing intermediate results to the output file
    with open(output_file, "a") as f:
        result_str = f'Round {round + 1}: {number_discard_points},{number_clusters_compression_set},{number_compression_points},{number_points_retained_set}\n'
        f.write(result_str)
        result_dict = {}

# Merge CS and DS clusters directly if their Mahalanobis distance < 2𝑑
for ds_cluster_id, ds_points in ds_cluster_points.items():
    result_dict.update({point: ds_cluster_id for point in ds_points})

for cs_cluster_id, cs_points in cs_cluster_points.items():
    result_dict.update({point: cs_cluster_id for point in cs_points})

if len(rs_outliers) > 0:
    rs_data = np_data_set[4][list(rs_outliers), 0]
    updated_rs_outliers = set([int(n) for n in rs_data])

# Assign -1 to the updated RS outliers or points not in clusters
result_dict.update({point: -1 for point in updated_rs_outliers})

with open(output_file, "a") as file:
    file.write('\n')
    file.write('The clustering results:\n')
    ordered_result = sorted(result_dict.keys(), key=int)
    for point in ordered_result:
        file.write(str(point) + ',' + str(result_dict[point]) + '\n')
