import os
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:27890"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:27890"
from fastapi import FastAPI
import uvicorn
from typing import List, Dict, Any  # Import type hints make the code clearer


from search_engine.milvus_search_engine import MilvusSearchEngine
from PIL import Image
import base64
from io import BytesIO

# Creating a FastAPI Application Instance
app = FastAPI(
    title="Hybrid Search Engine API",
    description="Provides search functionality using a pre-initialized HybridSearchEngine.",
    version="1.0.0",
)

# Global variable used to store HybridSearchEngine instances
search_engine: MilvusSearchEngine = None


# Initialize HybridSearchEngine when the application starts
@app.on_event("startup")
async def startup_event():
    global search_engine
    print("Initializing SearchEngine...")
    search_engine = MilvusSearchEngine(
        milvus_uri="http://localhost:19530",
        embed_model_name="vidore/colqwen2-v1.0",
        collection_name="pdf_datasets",
        device_str="cuda:0",
    )


# Define a hybrid search API endpoint (text + image)
@app.post(
    "/search",
    summary="Perform a hybrid search query with both text and images.",
    description="Executes a hybrid search using both text queries and base64-encoded images.",
    response_model=List[List[Dict[str, Any]]],
)
async def search(request_data: Dict[str, Any]):
    """
    Perform a mixed search operation (text + image).

    Args:
        request_data: A dictionary containing the query data, which may include 'queries' and 'images' fields.

    Returns:
        Search results list.
    """
    import json

    # Processing text queries
    queries = request_data.get("queries", [])
    if isinstance(queries, str):
        queries = [queries]

    # Optional partition
    partition = request_data.get("partition")

    # Processing base64 encoded images
    pil_images = None
    if "images" in request_data:
        images_data = request_data["images"]
        if isinstance(images_data, str):
            # If it's a JSON string, it needs to be parsed.
            try:
                images_data = json.loads(images_data)
            except:
                images_data = [images_data]

        if images_data:
            pil_images = []
            for img_base64 in images_data:
                if img_base64:
                    try:
                        # Decode base64 image
                        if img_base64.startswith("data:image"):
                            # Processing data URL format
                            img_data = base64.b64decode(img_base64.split(",")[-1])
                        else:
                            # Direct base64 encoding
                            img_data = base64.b64decode(img_base64)

                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(img_data))
                        pil_images.append(pil_image)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        continue

    # Perform a mixed search, passing a partition (ignore if None)
    results_batch = search_engine.batch_search(
        queries=queries, images=pil_images, partition=partition
    )

    # Formatting return
    results_batch = [
        [dict(idx=idx, image_file=file) for idx, file in enumerate(query_results)]
        for query_results in results_batch
    ]
    return results_batch


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


