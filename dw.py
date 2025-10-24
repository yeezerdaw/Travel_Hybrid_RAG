from pinecone import Pinecone, ServerlessSpec
import config

pc = Pinecone(api_key=config.PINECONE_API_KEY)
index_name = config.PINECONE_INDEX_NAME
if index_name in pc.list_indexes().names():
    print("Deleting existing index (data will be lost):", index_name)
    pc.delete_index(name=index_name)

print("Creating new index with dim=1024")
pc.create_index(
    name=index_name,
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)