import hashlib
import json
import time

import numpy as np
import torch
from pymilvus import MilvusClient
from tqdm import tqdm


def get_metadata(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line


def compute_embedding(abstract, embedding_model):
    start = time.time()
    res = embedding_model.encode(abstract)
    print("Embedding time: ", time.time() - start)
    return res


def get_embedding(abstract, embedding_model=None):

    if embedding_model is None:
        return torch.zeros(768)
    else:
        return compute_embedding(abstract, embedding_model)


def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


# Add hash to existing database entries
def compute_hash(entry):
    serializable_entry = convert_to_serializable(entry)  # Ensure JSON serialization compatibility
    json_string = json.dumps(serializable_entry, sort_keys=True)  # Serialize entry for hashing
    return hashlib.sha256(json_string.encode("utf-8")).hexdigest()  # Compute SHA-256 hash


# Assuming the database has been loaded into a collection


def create_hash_table_from_database(database_file, collection_name="abstracts", limit=1000):
    print("Creating hash table from database {}...".format(database_file))
    client = MilvusClient(database_file)
    hash_table = {}  # id -> hash

    print("Querying database of size", client.query(collection_name=collection_name, output_fields=["count(*)"]))
    offset = 0
    while True:
        query_start = time.time()
        entries = client.query(
            collection_name=collection_name, output_fields=["id", "hash"], filter="id != ''", offset=offset, limit=limit
        )
        if not entries:
            break
        for entry in entries:
            hash_table[entry["id"]] = entry["hash"]
        offset += limit
        print(f"Query time: {time.time()-query_start:.2f} query length: {len(entries)}, offset: {offset}")
    return hash_table


def create_hash_table_from_database_fast(database_file, collection_name="abstracts"):
    print("Creating hash table from database {}...".format(database_file))
    client = MilvusClient(database_file)
    hash_table = {}  # id -> hash

    print("Querying database of size", client.query(collection_name=collection_name, output_fields=["count(*)"]))
    iterator = client.query_iterator(collection_name=collection_name, output_fields=[id, hash], filter="id != ''")

    try:
        while True:
            start_time = time.time()
            # Large batch size to minimize iteration overhead
            batch = iterator.next(batch_size=10000)

            if not batch:
                break

            # Efficient dict update
            hash_table.update({entry["id"]: entry["hash"] for entry in batch})
            print(f"Processed batch: {len(batch)} entries in {time.time()-start_time:.2f} seconds")

    finally:
        iterator.close()

    return hash_table


def create_hash_table_from_dataset(dataset_file):
    print("Creating hash table from dataset {}...".format(dataset_file))
    hash_table = {}  # hash -> id
    total_items = sum(1 for _ in get_metadata(dataset_file))
    progress = tqdm(total=total_items, desc="Processing papers", unit="paper")
    for paper in tqdm(get_metadata(dataset_file)):
        try:
            paper = json.loads(paper)
        except Exception:
            print("Error loading paper")
            continue
        hash_key = compute_hash(paper)
        hash_table[paper["id"]] = hash_key
        progress.update(1)
    progress.close()
    return hash_table


if __name__ == "__main__":

    # Use the dataset loaded from Kaggle to build the hash table
    dataset_file = "/Users/yushixing/Cornell/arxiv-metadata-oai-init.json"
    hash_table_from_dataset = True

    # database_file = "/home/sy774/fall2024/database.db"
    # collection_name = "arxiv_meta"

    database_file = "/Users/yushixing/Cornell/database.db"
    collection_name = "abstracts"
    hash_table_from_database = False

    hash_table_file = "id_hash_table.json"

    if hash_table_from_database:
        id_hash_table = create_hash_table_from_database_fast(database_file, collection_name=collection_name)
        # id_hash_table = create_hash_table_from_database(database_file, collection_name=collection_name)
        json.dump(id_hash_table, open(hash_table_file, "w"))

    elif hash_table_from_dataset:
        id_hash_table = create_hash_table_from_dataset(dataset_file)
        json.dump(id_hash_table, open(hash_table_file, "w"))
