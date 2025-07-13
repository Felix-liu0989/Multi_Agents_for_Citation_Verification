import hashlib
import json
import sqlite3
import time

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


def hashing_database(dataset_file, database_file, embedding_model):
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS arxiv_metadata")

    # Create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS arxiv_metadata (
            id TEXT PRIMARY KEY,
            hash TEXT,
            abstract TEXT,
            embedding BLOB
        )
    """
    )

    # Insert data into the table
    for paper in get_metadata(dataset_file):
        paper = json.loads(paper)
        # Check if the entry already exists
        cursor.execute(
            """
            SELECT * FROM arxiv_metadata WHERE id = ?
        """,
            (paper["id"],),
        )
        entry = cursor.fetchone()
        # If the entry already exists,
        if cursor.fetchone():
            # check if the hash is the same
            if entry[1] != get_hash(paper["abstract"]):
                # calculate new hash, update hash and abstracts back to the database
                new_hash = get_hash(paper["abstract"])
                new_embedding = compute_embedding(paper["abstract"], embedding_model)
                cursor.execute(
                    """
                    UPDATE arxiv_metadata SET hash = ?, abstract = ?, embedding = ? WHERE id = ?
                """,
                    (new_hash, paper["abstract"], new_embedding, paper["id"]),
                )
                print(f"Updated entry for ID {paper['id']}.")

            print(f"Entry for ID {paper['id']} up-to-date. Skipping...")
            continue

        # Otherwise, insert a new entry
        new_entry = {}
        new_entry["hash"] = get_hash(paper["abstract"])
        new_entry["embedding"] = compute_embedding(abstract=paper["abstract"], embedding_model=embedding_model)
        new_entry["id"] = paper["id"]
        new_entry["abstract"] = paper["abstract"]

        cursor.execute(
            """
            INSERT OR REPLACE INTO arxiv_metadata (id, hash, abstract, embedding)
            VALUES (?, ?, ?, ?)
        """,
            (new_entry["id"], new_entry["hash"], new_entry["abstract"], new_entry["embedding"]),
        )
        print(f"Inserted new entry for ID {paper['id']}.")

    # Commit and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    dataset_file = "/home/sy774/fall2024/arxiv-metadata-oai-snapshot.json"
    database_file = "../database.db"

    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = topic_model.embedding_model.embedding_model
    hashing_database(dataset_file, database_file, embedding_model)
