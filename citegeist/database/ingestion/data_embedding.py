# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import hashlib
import json
import os
import sys
import tarfile
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
from bertopic import BERTopic

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = (
    "arxiv:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F612177%2F9995392%2Fbundle"
    "%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540"
    "kaggle-161607.iam.gserviceaccount.com%252F20241124%252Fauto%252Fstorage%252Fgoog4_request%26X-"
    "Goog-Date%3D20241124T060258Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-"
    "Signature%3D7a6425b9aa958a5ab304bcc62443bb05383d5a8c68f9b9af48f1ece25881901efec4ea26e663c21187"
    "08bd0c53103b035b6bfcdfe42796ef21649766888a429d49f043498f08b245ef955f07da887351f5cf2ae4f17826ca"
    "6f414c08e4600fad30f02a974d24ed6af9552c46ee6dbe6b731ae08c9a0dbbad062219e26dcbea283a893e269eb13d"
    "c1b3f8a0ade9a9431752fb26557dbd355ea915109706362796a2da03b1c9041a6557a5b7f677c6f0a102b70c31f94c"
    "773f29a530cebc6a05739dcfe2d9841a9bccc46836651e1601517faf9a49ba99cb26968ddfa94311ac0536dcc466ec"
    "51d8ea9e4e83578be8e44662b510a69fa845c03b9e7498b4ae5e22"
)

KAGGLE_INPUT_PATH = "arxivDrag/kaggle/input"
KAGGLE_WORKING_PATH = "arxivDrag/kaggle/working"
KAGGLE_SYMLINK = "arxivDrag/kaggle"
if not os.path.isdir("/home/cbb89/arxivDrag/kaggle"):
    os.makedirs(KAGGLE_SYMLINK)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777)
    os.makedirs(KAGGLE_WORKING_PATH, 0o777)

    for data_source_mapping in DATA_SOURCE_MAPPING.split(","):
        directory, download_url_encoded = data_source_mapping.split(":")
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = fileres.headers["content-length"]
                print(f"Downloading {directory}, {total_length} bytes compressed")
                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while len(data) > 0:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)
                if filename.endswith(".zip"):
                    with ZipFile(tfile) as zfile:
                        zfile.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name) as tarfile:
                        tarfile.extractall(destination_path)
                print(f"\nDownloaded and uncompressed: {directory}")
        except HTTPError:
            print(f"Failed to load (likely expired) {download_url} to path {destination_path}")
            continue
        except OSError:
            print(f"Failed to load {download_url} to path {destination_path}")
            continue

print("Data source import complete.")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+EnterX) will list all files under the input directory


for dirname, _, filenames in os.walk("arxivDrag/kaggle/input/arxiv/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Open the JSON file and load each line as a separate JSON object
with open(
    "arxivDrag/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json",
    "r",
    encoding="utf-8",
) as file:
    data = [json.loads(line) for line in file]

print(data[:10])  # Show the first 10 entries


# Load the BERTopic model
topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
embedding_model = topic_model.embedding_model.embedding_model

# Move the embedding model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = embedding_model.to(device)

# Settings
checkpoint_interval = 10000  # Save every 10,000 abstracts
batch_size = 256  # To avoid memory issues during encoding
output_file = "arxivDrag/processed_data_final_noAbstract.jsonl"  # JSONL format
checkpoint_file = "arxivDrag/checkpoint.json"

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r", encoding="utf-8") as chk_file:
        checkpoint_data = json.load(chk_file)
        last_processed_index = checkpoint_data.get("last_processed_index", 0)
else:
    last_processed_index = 0

# Ensure output file exists
if not os.path.exists(output_file):
    open(output_file, "w").close()  # Create an empty file


# Helper functions
def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def compute_hash(entry):
    serializable_entry = convert_to_serializable(entry)  # Ensure JSON serialization compatibility
    json_string = json.dumps(serializable_entry, sort_keys=True)  # Serialize entry for hashing
    return hashlib.sha256(json_string.encode("utf-8")).hexdigest()  # Compute SHA-256 hash


def process_batch(data_batch):
    abstracts = [entry["abstract"] for entry in data_batch if "abstract" in entry]
    embeddings = []
    topics = []

    for start_idx in range(0, len(abstracts), batch_size):
        sub_batch = abstracts[start_idx : start_idx + batch_size]
        sub_embeddings = embedding_model.encode(sub_batch, show_progress_bar=True)
        sub_topics, _ = topic_model.transform(sub_batch)

        embeddings.extend(sub_embeddings)
        topics.extend(sub_topics)

    # Create a structure to hold the results
    results = []
    for idx, entry in enumerate(data_batch):
        if "abstract" in entry:
            entry_hash = compute_hash(entry)  # Compute the hash from the original entry
            result = {
                "id": entry["id"],
                "embedding": embeddings[idx].tolist(),
                "topic": topics[idx],
                "hash": entry_hash,
            }
            results.append(convert_to_serializable(result))
    return results


# Process the dataset in batches
for i in range(last_processed_index, len(data), checkpoint_interval):
    batch = data[i : i + checkpoint_interval]

    # Process the batch and get embeddings and topics
    print(f"Processing batch {i // checkpoint_interval + 1} of {len(data) // checkpoint_interval + 1}...")
    processed_batch = process_batch(batch)

    # Append the processed batch to the output file in JSONL format
    with open(output_file, "a", encoding="utf-8") as out_file:
        for entry in processed_batch:
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")  # Each entry on a new line

    # Save the checkpoint
    last_processed_index = i + checkpoint_interval
    with open(checkpoint_file, "w", encoding="utf-8") as chk_file:
        json.dump({"last_processed_index": last_processed_index}, chk_file)

    print(f"Processed up to index {last_processed_index} abstracts.")

print("Finished writing to file")
