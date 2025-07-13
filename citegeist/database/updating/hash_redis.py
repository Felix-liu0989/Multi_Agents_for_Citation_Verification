import hashlib
import json
import time

import redis
from bertopic import BERTopic


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


def hashing_database(dataset_file, redis_host, redis_port, embedding_model):
    # Connect to Redis
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    # Insert data into Redis
    for paper in get_metadata(dataset_file):
        paper = json.loads(paper)
        paper_id = paper["id"]
        paper_hash = get_hash(paper["abstract"])

        # Check if the entry already exists by ID
        if r.exists(f"id:{paper_id}"):
            # Retrieve the existing hash
            existing_hash = r.hget(f"id:{paper_id}", "hash")
            if existing_hash != paper_hash:
                # Update the hash, abstract, and embedding
                new_embedding = compute_embedding(paper["abstract"], embedding_model)
                r.hset(
                    f"id:{paper_id}",
                    mapping={"hash": paper_hash, "abstract": paper["abstract"], "embedding": new_embedding},
                )
                print(f"Updated entry for ID {paper_id}.")
            else:
                print(f"Entry for ID {paper_id} up-to-date. Skipping...")
            continue

        # Otherwise, insert a new entry
        new_embedding = compute_embedding(paper["abstract"], embedding_model)
        r.hset(f"id:{paper_id}", mapping={"hash": paper_hash, "abstract": paper["abstract"], "embedding": new_embedding})
        print(f"Inserted new entry for ID {paper_id}.")


if __name__ == "__main__":
    dataset_file = "/home/sy774/fall2024/arxiv-metadata-oai-snapshot.json"
    redis_host = "localhost"
    redis_port = 6379

    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = topic_model.embedding_model.embedding_model
    hashing_database(dataset_file, redis_host, redis_port, embedding_model)
