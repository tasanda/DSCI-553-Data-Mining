import random, sys
from blackbox import BlackBox

#input_filename = 'users.txt'
#stream_size = 100
#num_of_asks = 30
#output_filename = 'task3output.txt'
input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

random.seed(553)
fixed_size_list = []
seqnum = 0
bx = BlackBox()
# The goal of task3 is to implement the fixed-size sampling method (Reservoir Sampling Algorithm).
# we assume that the memory can only save 100 users, so we need to use the fixed size sampling method to only keep
# part of the users as a sample in the streaming. When the streaming of the users comes, for the first 100 users, you can directly
# save them in a list.
def resevoir_sampling(stream_user):
    global seqnum
    for user_id in stream_user:
        # For the probability of whether to accept a sample or discard, use random.random() which generates
        # a floating point number between 0 and 1. If this randomly generated probability is less than s/n, we accept the sample
        seqnum += 1
        if len(fixed_size_list) < 100:
            fixed_size_list.append(user_id)
        elif random.random() < 100 / seqnum:
            replacement_position = random.randint(0, 99)
            fixed_size_list[replacement_position] = user_id
            # accept it we need to find an index in the array for replacement. For this purpose use random.randInt()
            # with appropriate boundaries to generate an index into the array and use this for replacement of the sample.
        if seqnum % 100 == 0:
            user_0_id = fixed_size_list[0]
            user_20_id = fixed_size_list[20]
            user_40_id = fixed_size_list[40]
            user_60_id = fixed_size_list[60]
            user_80_id = fixed_size_list[80]
    return user_0_id, user_20_id, user_40_id, user_60_id, user_80_id, seqnum

with open(output_filename, 'w') as f:
    f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    for i in range(num_of_asks):
        stream_users_next = bx.ask(input_filename, stream_size)
        user_0_id, user_20_id, user_40_id, user_60_id, user_80_id, seqnum = resevoir_sampling(stream_users_next)
        f.write(f"{seqnum},{user_0_id},{user_20_id},{user_40_id},{user_60_id},{user_80_id}\n")