from blackbox import BlackBox
import random
import binascii
import csv
import sys
import time
import hashlib

def myhashs(s):
    hash_functions = []
    for i in range(num_of_hash):
        a = random.randint(1, 9000000000)
        b = random.randint(1, 9000000000)
        M = 997
        P = 1e9 + 7

        def make_hash(x, a=a, b=b, P=P, M=M):
            return ((a * x + b) % P) % M

        hash_functions.append(make_hash)

    result = []
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    for func in hash_functions:
        result.append(func(x))
    return result

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]
# input_filename = 'users.txt'
# stream_size = 300
# num_of_asks = 30
# output_filename = 'task2_output.txt'

num_of_hash = 100
start_time = time.time()
bx = BlackBox()


def estimate_counts(stream_users):
    unique_users = set()
    hash_values = []

    # Generate hash values for each user and collect unique users
    for user in stream_users:
        hashed_result = myhashs(user)
        unique_users.add(user)
        hash_values.append(hashed_result)

    total_trailing_zeros = 0
    for i in range(num_of_hash):
        max_trailing_zeros = 0

        # Calculate trailing zeros for each hash value
        for value in hash_values:
            binary_value = bin(int(value[i]))[2:]
            trailing_zero_count = len(binary_value) - len(binary_value.rstrip('0'))
            max_trailing_zeros = max(max_trailing_zeros, trailing_zero_count)

        total_trailing_zeros += 2 ** max_trailing_zeros

    # Calculate estimation
    estimation = total_trailing_zeros // num_of_hash

    return estimation

with open(output_filename, 'w') as f:
    f.write("Time,Ground Truth,Estimation\n")
    ground_truth_sum = 0  # Variable to store the sum of ground truths
    for i in range(num_of_asks):
        stream_users_next = bx.ask(input_filename, stream_size)
        ground_truth = len(set(stream_users_next))
        result = estimate_counts(stream_users_next)
        f.write(f"{i},{ground_truth},{result}\n")
end_time = time.time()
print(f"Duration: {end_time - start_time}")