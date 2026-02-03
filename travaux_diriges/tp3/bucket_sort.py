from mpi4py import MPI
import numpy as np
import time

def bucket_sort(arr, num_buckets=None):     # IA made sort algorithm for time comparison

    if len(arr) == 0:
        return [], 0

    start = time.time()

    if num_buckets is None:
        num_buckets = len(arr)
    buckets = [[] for _ in range(num_buckets)]

    for value in arr:
        index = int(value * num_buckets)  # value in [0,1)
        if index >= num_buckets:          # edge case for value == 1
            index = num_buckets - 1
        insert_value(buckets[index], value) 

    sorted_arr = []
    for bucket in buckets:
        for value in bucket:
            sorted_arr.append(value)
    end = time.time()
    return sorted_arr, end-start


def insert_value(array, value):
    for i in range(len(array)):
        if value < array[i]:
            array.insert(i, float(value))
            return
    array.append(value)
        
    
def generate_array(size=10):
    random_array = np.floor(np.random.rand(size) * 100) / 100
    return random_array

def bucket_sort_parallel(data):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    np = comm.Get_size()
    
    array_size = 1 / np     # floor values for every size
    temp = []               # temporary array to be sorted
    
    if rank == 0:
        start = time.time()
        # print("List: ", data)
    else:
        data = None
    
    data = comm.bcast(data, root=0)

    for value in data:
        # partitioning each value to the respective bucket
        bucket_id = int((value-0)/array_size)
        bucket_id = min(bucket_id, np - 1) # for the case where bucket_id = np
            
        # print("Value ", value, " going to bucket ", bucket_id)
        
        if bucket_id == rank:
            insert_value(temp, value)
    
    temp = [float(x) for x in temp]
    result = comm.gather(temp, root = 0)
    if rank == 0:
        end = time.time()
        sorted_array = [x for sublist in result for x in sublist]
        return sorted_array, end-start
    else:
        return [0], 0          
            

if __name__ == "__main__":
    size = 1_000
    data = generate_array(size)
    sorted, time_ref = bucket_sort(data)
    sorted_par, time_p = bucket_sort_parallel(data)
    if time_p:
        time_par = time_p

        # print("Previous array: ", data)
        # print("Sorted array parallel: ", sorted_par)
        print("Time passed parallel:", time_par)
        # print("\n\nSorted array reference: ", sorted)
        print("Time passed reference:", time_ref)
    