import os
import json
import random
import torch
from collections import defaultdict
from itertools import count

#Path
raw_file_path = "reviews.json"
processed_file_path = ["data_train.pt","data_test.pt","data_val.pt"]

def process_data(raw_file_path=raw_file_path):
    # Check first if the data is already there
    with open(raw_file_path,'r') as file:
        reviews = json.load(file)

    COO_format = [[], []]  # Rows, Columns, Data

    # Use defaultdict with itertools.count to auto-increment indices
    node_mapping = defaultdict(count().__next__)

    for review in reviews:
        user_id = review['user_id']
        item_id = review['business_id']
        if review['stars'] >= 3:

            # Append to COO_format
            COO_format[0].append(node_mapping[user_id])  # Row index (user)
            COO_format[1].append(node_mapping[item_id])  # Column index (item)

    edge_index = torch.tensor(COO_format, dtype=torch.long)

    return edge_index

def split_data(edge_index,train_size=0.7,test_size=0.1):

    #Size if splits
    N = edge_index.size(1)
    N_train = int(N * train_size)
    N_test = int(N * test_size)

    # Shuffle the data before splitting
    indices = torch.randperm(N)
    shuffled_edge_index = edge_index[:, indices]

    # Perform the splits
    edge_index_train = shuffled_edge_index[:, :N_train]
    edge_index_test = shuffled_edge_index[:, N_train:N_train + N_test]
    edge_index_val = shuffled_edge_index[:, N_train + N_test:]

    return edge_index_train,edge_index_test,edge_index_val

def make_contiguous(ids):
    # Flatten the input tensor and find unique values
    unique_vals = torch.unique(ids)

    # Create a mapping from old IDs to new contiguous IDs
    mapping = torch.zeros(ids.max() + 1, dtype=torch.long)
    mapping[unique_vals] = torch.arange(len(unique_vals), dtype=torch.long)

    # Apply the mapping to the input tensor
    reindexed_ids = mapping[ids]

    return reindexed_ids

def get_data():
    #Check if the processed data already exists and create it if not
    if not all([os.path.exists(os.path.join('src/data/processed',path)) for path in processed_file_path]):
        edge_index = process_data(raw_file_path=raw_file_path)
        edge_index_train,edge_index_test,edge_index_val = split_data(edge_index=edge_index)

        # Save edge_index as a .pt file
        torch.save(make_contiguous(edge_index_train), 'src/data/processed/data_train.pt')
        torch.save(make_contiguous(edge_index_test), 'src/data/processed/data_test.pt')
        torch.save(make_contiguous(edge_index_val), 'src/data/processed/data_val.pt')
    else:
        #load data if it already exists
        edge_index_train = torch.load('src/data/processed/data_train.pt')
        edge_index_test = torch.load('src/data/processed/data_test.pt')
        edge_index_val = torch.load('src/data/processed/data_val.pt')

    return {
        'train':edge_index_train,
        'test':edge_index_test,
        'val':edge_index_val
        }

def create_random_batches(N, batch_size):
    elements = list(range(1, N + 1))
    random.shuffle(elements)  # Shuffle elements to randomize order
    batches = [elements[i:i + batch_size] for i in range(0, len(elements), batch_size)]
    return batches

def get_infos(edge_index):
    n_users,n_items = edge_index[0].unique().size(0),edge_index[1].unique().size(0)
    print(f'Number of users : {n_users}\nNumber of items : {n_items}')
    return n_users,n_items
    
            