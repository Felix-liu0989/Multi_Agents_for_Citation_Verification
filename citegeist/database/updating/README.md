# Hash Functions

This folder contains implementations of various hash functions. 

## Main Contents

- `hash_tabling.py`: Run this to create a hash table
    - Change `database_file = "/home/sy774/fall2024/database.db"` to your own file of database path.
    - Change `hash_table_file = "/home/sy774/id_hash_table.json"`  to the path you want to save the hash table
- `hash_reload.py`: After running `hash_tabling.py`, run this to reload and update the arxiv data. Remember to change `hash_table_file = "/home/sy774/id_hash_table.json"`
- `hash_merged.py`: A merged version of the previous two functionality.





## Quick use of hash reload

```python

from citegeist import hash_reload
import kagglehub

# Download updated arxiv data
dataset_file = kagglehub.dataset_download("Cornell-University/arxiv") + "/arxiv-metadata-oai-snapshot.json"
print("Path to dataset files:", dataset_file)

database_file = "/Users/yushixing/Cornell/database.db"
collection_name = "abstracts"
hash_table_file = "/Users/yushixing/Cornell/id_hash_table.json"

# Load hash table
id_hash_table = json.load(open(hash_table_file, "r"))

# Reload and lookup
updated_hash_table = hash_reload.reload_and_lookup(dataset_file, database_file, id_hash_table,
                                                   collection_name=collection_name)
json.dump(updated_hash_table, open(hash_table_file, "w"))

```





## Other contents:

- `hash_redis.py`: Redis version of hash table
- `hash_sqlite.py`: sqlite version of hash table`hash_table_file = "/home/sy774/id_hash_table.json"`
- `hash.py`: pymilvus version of hash table
