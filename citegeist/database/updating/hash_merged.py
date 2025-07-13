import hashlib
import json
import time

import torch
from bertopic import BERTopic
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


# Add hash to existing database entries
def compute_hash(entry: dict) -> str:
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


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


###
# Then reload the database from kaggle, and for every metadata object compute the hash and do a lookup based on this.
# If the entry already exists pass. If the entry does not exist, do another lookup based on the id and determine whether
# the abstract has changed.
# If not pass. If yes, replace the existing row (this includes recomputing the embedding).
# If the Id is also not found insert the entry as a new row (also includes computing the embedding).
# You can look for examples of this in the embeddingBase.py and the small database.
# Updating hash if the paper has changed but the id is there.
###


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
    for paper in tqdm(get_metadata(kaggle_path)):

        new_hash_key = hashlib.sha256(paper.encode("utf-8")).hexdigest()
        # Search for the hash in the database
        paper = json.loads(paper)
        paper_id = paper["id"]
        old_hash_key = hash_table.get(paper_id, 0)

        # Case 0: Hashes are the same, abstract is the same
        if old_hash_key == new_hash_key:
            continue

        # Case 1: Hash is different, but ID is found, update database and hash table
        if old_hash_key:
            # Update entry: replace abstract and recompute hash and embedding
            updated_entry = {}
            updated_entry["hash"] = new_hash_key
            updated_entry["embedding"] = get_embedding(abstract=paper["abstract"], embedding_model=embedding_model)
            updated_entry["id"] = paper["id"]
            updated_entry["abstract"] = paper["abstract"]
            # updated_entry["topic"] = existing_entry["topic"]
            client.delete(collection_name=collection_name, filter=f'id == "{paper_id}"')
            client.insert(collection_name=collection_name, data=[updated_entry])
            hash_table[paper_id] = new_hash_key

            import pdb

            pdb.set_trace()

        # Case 2: id not found, insert new row
        else:
            new_entry = {}
            new_entry["hash"] = new_hash_key
            new_entry["embedding"] = get_embedding(abstract=paper["abstract"], embedding_model=embedding_model)
            new_entry["id"] = paper["id"]
            new_entry["abstract"] = paper["abstract"]
            # new_entry["topic"] = topic_model.transform([paper["abstract"]])[0]
            client.insert(collection_name=collection_name, data=[new_entry])
            hash_table[paper_id] = new_hash_key
            print(f"Inserted new entry for ID {paper['id']}.")

    print("Database update completed.")


if __name__ == "__main__":

    # Use the dataset loaded from Kaggle to build the hash table
    dataset_file = "/home/sy774/fall2024/arxiv-metadata-oai-snapshot.json"
    hash_table_from_dataset = False

    database_file = "/home/sy774/fall2024/database.db"
    collection_name = "arxiv_meta"
    hash_table_from_database = True

    hash_table_file = "/home/sy774/id_hash_table.json"

    id_hash_table = None
    if database_file:
        id_hash_table = create_hash_table_from_database(database_file, collection_name=collection_name)
        json.dump(id_hash_table, open(hash_table_file, "w"))
    elif hash_table_from_dataset:
        id_hash_table = create_hash_table_from_dataset(dataset_file)
        json.dump(id_hash_table, open(hash_table_file, "w"))

    if not id_hash_table:
        id_hash_table = json.load(open(hash_table_file, "r"))

    # Reload and lookup
    reload_and_lookup(dataset_file, database_file, id_hash_table, collection_name=collection_name)
