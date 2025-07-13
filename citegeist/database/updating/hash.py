import hashlib
import json
import time

import torch
from bertopic import BERTopic
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm

EMBEDDING_PATH = "../../../../database_mini.db"
HASH_TABLE_CREATED = True


def get_metadata(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line


def compute_embedding(abstract, embedding_model):
    start = time.time()
    res = embedding_model.encode(abstract)
    print("Embedding time: ", time.time() - start)
    return res


def get_embedding(id, abstract, embedding_model=None):
    # mini_client = MilvusClient(EMBEDDING_PATH)
    # result = mini_client.query(
    #     collection_name="abstracts",
    #     filter=f'id == "{id}"',  # Use 'filter' instead of 'expr'
    #     output_fields=["embedding"]
    # )
    # if not result:
    #     if embedding_model is None:
    #         return torch.zeros(128)
    #     else:
    #         return compute_embedding(abstract, embedding_model)
    # else:
    #     return result[0]["embedding"]

    if embedding_model is None:
        return torch.zeros(128)
    else:
        return compute_embedding(abstract, embedding_model)


# Add hash to existing database entries
def compute_hash(entry: dict) -> str:
    """Compute a SHA256 hash for a given entry."""
    entry_str = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


# Assuming the database has been loaded into a collection
def create_hash_table(collection_name: str):
    client = MilvusClient(EMBEDDING_PATH)
    hash_table = {}
    for entry in client.query(collection_name=collection_name, output_fields=["id", "hash", "embedding"]):
        hash_table[entry["hash"]] = entry
    return


def add_hash_to_database(dataset_file, client):

    total_items = sum(1 for _ in get_metadata(dataset_file))
    progress = tqdm(total=total_items, desc="Processing papers", unit="paper")
    i = 0
    for paper in tqdm(get_metadata(dataset_file)):
        i += 1
        if i >= 10000:
            break
        hash_key = hashlib.sha256(paper.encode("utf-8")).hexdigest()
        paper = json.loads(paper)
        id = paper["id"]
        # print(id, hash_key)
        # Get embedding from client mini
        embedding = get_embedding(id, abstract=paper["abstract"])
        # for k, v in json.loads(paper).items():
        #     print(k, v)
        # Add the hash key and metadata to the original database under the column hash based
        # on the id of the paper
        client.insert(
            collection_name="abstracts",
            data={"id": id, "hash": hash_key, "abstract": paper["abstract"], "embedding": embedding},
        )
        progress.update(1)
    progress.close()


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
def reload_and_lookup(dataset_file, database_file):
    # Load the BERTopic model
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = topic_model.embedding_model.embedding_model

    # Move the embedding model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = embedding_model.to(device)

    # Get the latest version of the dataset
    # import kagglehub
    # kaggle_path = kagglehub.dataset_download("Cornell-University/arxiv", path="..") +
    # "/arxiv-metadata-oai-snapshot.json"
    # print("Path to dataset files:", kaggle_path)
    kaggle_path = dataset_file
    client = MilvusClient(database_file)

    # write tqdm to show the progress of the loop by showing the total length and expected finish time
    for paper in tqdm(get_metadata(kaggle_path)):

        hash_key = hashlib.sha256(paper.encode("utf-8")).hexdigest()
        # Search for the hash in the database
        paper = json.loads(paper)
        existing_hash = client.query(
            collection_name="abstracts", filter=f'hash == "{hash_key}"', output_fields=["id", "hash", "abstract"]
        )
        # Case 1: Entry with hash already exists
        if len(existing_hash) > 0:
            continue

        existing_id = client.query(
            collection_name="abstracts", filter=f'id == "{paper["id"]}"', output_fields=["id", "hash", "abstract"]
        )
        if len(existing_id) > 0:
            # Case 2: Hash doesn't exist but ID is found

            existing_entry = existing_id[0]
            if existing_entry["abstract"] != paper["abstract"]:
                # Update entry: replace abstract and recompute hash and embedding
                updated_entry = paper.copy()
                updated_entry["hash"] = hash_key
                updated_entry["embedding"] = get_embedding(
                    paper["id"], abstract=paper["abstract"], embedding_model=embedding_model
                )
                updated_entry["id"] = paper["id"]
                updated_entry["abstract"] = paper["abstract"]
                # updated_entry["topic"] = existing_entry["topic"]

                client.update(collection_name="abstracts", data=[updated_entry])
                print(f"Updated entry for ID {paper['id']}.")
            continue

        import pdb

        pdb.set_trace()

        # Case 3: ID not found; insert new row
        new_entry = paper.copy()
        new_entry["hash"] = hash_key
        new_entry["embedding"] = get_embedding(paper["id"], abstract=paper["abstract"], embedding_model=embedding_model)
        new_entry["id"] = paper["id"]
        new_entry["abstract"] = paper["abstract"]
        # new_entry["topic"] = topic_model.transform([paper["abstract"]])[0]

        client.insert(collection_name="abstracts", data=[new_entry])
        print(f"Inserted new entry for ID {paper['id']}.")


def hashing_database(dataset_file, database_file):
    client = MilvusClient(database_file)
    # create the database with the hash column and the other metadata of the paper and embedding

    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True)
    hash_field = FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=255)
    abstract_field = FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=1024)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)

    # Create collection schema
    schema = CollectionSchema(fields=[id_field, hash_field, abstract_field, embedding_field], auto_id=False)

    # Create collection
    client.create_collection(collection_name="abstracts", schema=schema)
    add_hash_to_database(dataset_file, client)


if __name__ == "__main__":
    dataset_file = "/home/sy774/fall2024/arxiv-metadata-oai-snapshot.json"
    database_file = "../database.db"

    if not HASH_TABLE_CREATED:
        hashing_database(dataset_file, database_file)

    # Reload and lookup
    reload_and_lookup(dataset_file, database_file)
