import os
import sys

from pymilvus.bulk_writer import bulk_import

"""
Usage:
python data_ingest.py <file1> [file2] ...
"""

assert len(sys.argv) >= 2

response = bulk_import(
    url=os.environ["MILVUS_URI"],
    collection_name="abstracts",
    files=[[f"{file}"] for file in sys.argv[1:]],
)

job_id = response.json()["data"]["job_id"]
print(f"Started import under Job ID: {job_id}")
