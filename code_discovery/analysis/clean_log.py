import yaml
from pymongo import MongoClient


def clean_mm_rag_collection(mongo_config_path, collection_name):
    # Load MongoDB configuration
    with open(mongo_config_path, "r") as f:
        mongo_config = yaml.safe_load(f)

    # Connect to MongoDB
    mongo_url = mongo_config["mongo"]["url"]
    client = MongoClient(mongo_url)
    collection = client["emo"][collection_name]

    # Define the conditions to filter out unwanted documents
    query = {
        "surrogate_ratio": 3,
    }

    # Perform the deletion based on the query
    delete_result = collection.delete_many(query)
    print(
        f"Deleted {delete_result.deleted_count} documents based on the cleaning criteria."
    )


if __name__ == "__main__":
    mongo_config_path = "config.yaml"
    collection_name = "mm_rag"
    clean_mm_rag_collection(mongo_config_path, collection_name)
