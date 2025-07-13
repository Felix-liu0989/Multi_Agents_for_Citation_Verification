import hashlib
import json
import logging
import time

import numpy as np
import torch
from bertopic import BERTopic
from pymilvus import MilvusClient
from tqdm import tqdm

# Suppress BERTopic output
logging.getLogger("bertopic").setLevel(logging.WARNING)


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


def create_hash_table_from_database(database_file, collection_name="abstracts"):
    print("Creating hash table from database {}...".format(database_file))
    client = MilvusClient(database_file)
    hash_table = {}  # id -> hash
    query_start = time.time()
    entries = client.query(collection_name=collection_name, output_fields=["id", "hash", "embedding"], filter="id != ''")
    print(f"Query time: {time.time()-query_start:.2f} query length: {len(entries)}")
    for entry in tqdm(entries, desc="Processing entries", unit="entry"):
        hash_table[entry["id"]] = entry["hash"]
    return hash_table


def create_hash_table_from_dataset(dataset_file):
    print("Creating hash table from dataset {}...".format(dataset_file))
    hash_table = {}  # hash -> id
    total_items = sum(1 for _ in get_metadata(dataset_file))
    progress = tqdm(total=total_items, desc="Processing papers", unit="paper")
    for paper in tqdm(get_metadata(dataset_file)):
        paper = json.loads(paper)
        hash_key = compute_hash(paper)
        hash_table[paper["id"]] = hash_key
        progress.update(1)
    progress.close()
    return hash_table


# Reload database from Kaggle
def reload_and_lookup(dataset_file, database_file, hash_table, collection_name="abstracts"):
    # Load the BERTopic model
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = topic_model.embedding_model.embedding_model

    # Move the embedding model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = embedding_model.to(device)

    # Get the latest version of the dataset
    # import kagglehub
    # kaggle_path = kagglehub.dataset_download("Cornell-University/arxiv", path="..")
    # + "/arxiv-metadata-oai-snapshot.json"
    # print("Path to dataset files:", kaggle_path)
    kaggle_path = dataset_file
    client = MilvusClient(database_file)

    # write tqdm to show the progress of the loop by showing the total length and expected finish time

    update_ids = 0
    inserted_ids = 0
    for paper in tqdm(get_metadata(kaggle_path)):
        # Search for the hash in the database
        start = time.time()
        try:
            paper = json.loads(paper)
        except Exception:
            print("Error loading paper")
            continue
        new_hash_key = compute_hash(paper)
        paper_id = paper["id"]
        old_hash_key = hash_table.get(paper_id, 0)
        print(f"Hash computation time: {time.time()-start}")

        # Case 0: Hashes are the same, abstract is the same
        if old_hash_key == new_hash_key:
            continue

        # Case 1: Hash is different, but ID is found, update database and hash table
        if old_hash_key:
            # Update entry: replace abstract and recompute hash and embedding

            updated_entry = {}
            updated_entry["hash"] = new_hash_key
            updated_entry["embedding"] = get_embedding(abstract=paper["abstract"], embedding_model=embedding_model)
            print(f"Embedding computation time: {time.time()-start}")
            updated_entry["id"] = paper["id"]
            # updated_entry["abstract"] = paper["abstract"]
            updated_entry["topic"] = topic_model.transform([paper["abstract"]])[0].item()
            # client.update(collection_name=collection_name, data=[updated_entry], filter=f'id == "{paper_id}"')

            update_ids += 1
            # client.delete(collection_name=collection_name, filter=f'id == "{paper_id}"')
            client.insert(collection_name=collection_name, data=[updated_entry])
            print(f"Entry insertion time: {time.time()-start}")
            hash_table[paper_id] = new_hash_key
            print(f"Updated entry for ID {paper['id']}.")

            # import pdb; pdb.set_trace()

        # Case 2: id not found, insert new row
        else:
            new_entry = {}
            new_entry["hash"] = new_hash_key
            new_entry["embedding"] = get_embedding(abstract=paper["abstract"], embedding_model=embedding_model)
            new_entry["id"] = paper["id"]
            # new_entry["abstract"] = paper["abstract"]

            inserted_ids += 1
            new_entry["topic"] = int(topic_model.transform([paper["abstract"]])[0].item())
            client.insert(collection_name=collection_name, data=[new_entry])
            hash_table[paper_id] = new_hash_key
            print(f"Inserted new entry for ID {paper['id']}.")

    print(f"Database update completed.{update_ids} entries updated, {inserted_ids} entries inserted")
    return hash_table


def test():
    client = MilvusClient("/Users/yushixing/Cornell/database.db")
    entry = client.query(
        collection_name="abstracts", output_fields=["id", "hash", "embedding"], filter="id == '2412.00036'"
    )
    print(entry)


if __name__ == "__main__":

    # Use the dataset loaded from Kaggle to build the hash table
    dataset_file = "/Users/yushixing/Cornell/arxiv-metadata-oai-snapshot.json"

    # import kagglehub
    # dataset_file = kagglehub.dataset_download("Cornell-University/arxiv", path="/Users/yushixing/Cornell/") +
    # "/arxiv-metadata-oai-snapshot.json"
    # print("Path to dataset files:", dataset_file)
    hash_table_from_dataset = False

    database_file = "/Users/yushixing/Cornell/database.db"
    collection_name = "abstracts"
    hash_table_from_database = False

    hash_table_file = "/Users/yushixing/Cornell/id_hash_table.json"

    id_hash_table = json.load(open(hash_table_file, "r"))
    if not id_hash_table:
        if hash_table_from_dataset:
            id_hash_table = create_hash_table_from_dataset(dataset_file)
            json.dump(id_hash_table, open(hash_table_file, "w"))
        elif hash_table_from_database:
            id_hash_table = create_hash_table_from_database(database_file)
            json.dump(id_hash_table, open(hash_table_file, "w"))

    # Reload and lookup
    updated_hash_table = reload_and_lookup(dataset_file, database_file, id_hash_table, collection_name=collection_name)
    json.dump(updated_hash_table, open(hash_table_file, "w"))
