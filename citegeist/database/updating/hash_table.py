import hashlib
import json
import time

from bertopic import BERTopic

EMBEDDING_PATH = "../../../../database_mini.db"
HASH_TABLE_CREATED = True

# Initialize the id table
id_table = {}


def get_metadata(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line


def compute_embedding(abstract, embedding_model):
    start = time.time()
    res = embedding_model.encode(abstract)
    print("Embedding time: ", time.time() - start)
    return res


def get_hash(abstract):
    return hashlib.sha256(abstract.encode()).hexdigest()


def arxiv_update(id, abstract, embedding_model):
    abstract_hash = get_hash(abstract)

    # Try to locate using ID
    if id in id_table:
        old_hash = id_table[id]["hash_key"]
        if abstract_hash != old_hash:
            id_table[id]["hash_key"] = abstract_hash
            id_table[id]["abstract"] = abstract
        return id_table[id]

    # If not found, compute the embedding
    embedding = compute_embedding(abstract, embedding_model)

    # Store in both hash_table and id_table
    id_table[id] = {"hash": abstract_hash, "embedding": embedding, "abstract": abstract, "hash_key": abstract_hash}

    return id_table[id]


# build the hash table and id table
def update_hash_table(embedding_model):
    metadata = get_metadata("metadata.json")
    for line in metadata:
        paper = json.loads(line)
        arxiv_update(paper["id"], paper["abstract"], embedding_model)
    json.dump(id_table, open("id_table.json", "w"))


def lookup_hash_table(id, abstract, embedding_model):
    # abstract_hash = get_hash(abstract)

    if id in id_table:
        return id_table[id]

    # If not found, compute the embedding
    embedding = compute_embedding(abstract, embedding_model)

    return embedding


if __name__ == "__main__":
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = topic_model.embedding_model.embedding_model
    if HASH_TABLE_CREATED:
        update_hash_table(embedding_model)
    else:
        lookup_hash_table("2309.11101", "This is a test", embedding_model)
