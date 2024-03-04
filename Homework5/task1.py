import binascii, random, sys
from blackbox import BlackBox
import hashlib

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

# Constants
M = 69997  # number of bits in array
P = 99991  # A prime number larger than 'M'
NUM_HASH_FUNCS = 100  # Number of hash functions

# Create a global filter list with a length of M
global_filter = [0] * M  # Initialize all elements to 0 (False)

# Example hash functions
def hash_function_1(user_id, a, b):
    hash_input = str(user_id) + str(a) + str(b)
    return int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % P % M
def hash_function_2(user_id, a, b):
    hash_input = str(user_id) + str(a) + str(b)
    return int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % P % M

# List of hash functions
hash_function_list = [hash_function_1, hash_function_2]

def myhashs(s):
    result = []
    for hash_function in hash_function_list:
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        result.append(hash_function(s, a, b))
    return result

bx = BlackBox()
previous_user_set = bx.ask(input_filename, stream_size)

# Building the filter
for user_id in previous_user_set:
    user_id = int(binascii.hexlify(user_id.encode('utf8')), 16)
    hash_values = myhashs(user_id)
    for hash_value in hash_values:
        global_filter[hash_value] = 1

# print("Stream Users:", previous_user_set)
# print("Global Filter:", global_filter)

# Application: Check if new object o’ is in S
def bloom_filter_apply(stream_users_next):
    count_data_batch = 0
    for user_id in stream_users_next:
        user_id = int(binascii.hexlify(user_id.encode('utf8')), 16)
        hash_values = myhashs(user_id)
        if all(global_filter[hash_value] == 1 for hash_value in hash_values):
            #compare current hash value and user id and see if it's actually the same or a false positive
            if user_id not in previous_user_set:
                count_data_batch += 1
    FPR = count_data_batch/len(stream_users_next)
    # print("false positive rate in the batch:", FPR)
    return FPR


with open(output_filename, 'w') as f:
    f.write("Time,FPR\n")
    for i in range(num_of_asks):
        stream_users_next = bx.ask(input_filename, stream_size)
        result = bloom_filter_apply(stream_users_next)
        f.write(f"{i},{result}\n")
