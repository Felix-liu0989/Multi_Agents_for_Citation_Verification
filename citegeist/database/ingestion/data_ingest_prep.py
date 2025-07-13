import os.path
import sys

import jsonlines
from pymilvus import DataType, MilvusClient
from pymilvus.bulk_writer import BulkFileType, RemoteBulkWriter

"""
Usage:
python data_ingest_prep.py <path to source data (.jsonl)>

Please also define the two env variables MILVUS_URI (http://...) and MILVUS_TOKEN (user:pass)
"""
# Input validation
assert len(sys.argv) == 2
assert os.path.isfile(sys.argv[1])
assert sys.argv[1].endswith(".jsonl")

# Initiate Milvus Client
client = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ["MILVUS_TOKEN"])

# Define Milvus DB Schema
schema = client.create_schema(auto_id=False)
schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=50, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="hash", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="topic", datatype=DataType.INT64)

# Add index on the embedding vector values
index_params = MilvusClient.prepare_index_params()
index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="FLAT")

# Create Milvus collection based on schema
client.create_collection(collection_name="abstracts", schema=schema, index_params=index_params)

# Initialize RemoteBulkWriter
connect_param = RemoteBulkWriter.S3ConnectParam(
    endpoint=os.environ["MILVUS_MINIO_ENDPOINT"],
    access_key=os.environ["MILVUS_MINIO_ACCESS_KEY"],
    secret_key=os.environ["MILVUS_MINIO_SECRET_KEY"],
    bucket_name=os.environ["MILVUS_MINIO_BUCKET_NAME"],
    secure=False,
)
writer = RemoteBulkWriter(schema=schema, connect_param=connect_param, remote_path="/", file_type=BulkFileType.PARQUET)

# Read in the entire jsonl file data and insert it into the DB
count = 0
# Optionally, we can also filter to only include certain keys.
# accepted_keys = [79, 77, 76, 63, 62, 60, 40, 30, 19, 16]
with jsonlines.open(sys.argv[1]) as reader:
    for obj in reader:
        # if obj["topic"] not in accepted_keys:
        #     continue
        if count % 10000 == 0:
            writer.commit()
            print(f"Processed & committed {count} entries")
        writer.append_row({"id": obj["id"], "embedding": obj["embedding"], "hash": obj["hash"], "topic": obj["topic"]})
        count += 1

print(f"Finished processing {count} entries")
print(writer.batch_files)
writer.commit()
client.close()
