import concurrent.futures
import os
from typing import List, cast

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from pymilvus import MilvusClient, DataType
from pymilvus import connections, Collection
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_fusion.time_util import timeit

FAST_MODE = True


class MilvusColQwen:
    def __init__(self, milvus_client, collection_name, dim=128, alias="default"):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        self.dim = dim
        try:
            connections.get_connection(alias)
        except Exception:
            connections.connect(alias=alias, host="localhost", port="19530")
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"✅ Collection '{self.collection_name}' has been created and loaded successfully.")
            self.client.load_collection(collection_name=self.collection_name)
            self.collection = Collection(self.collection_name)
        else:
            print(f"⚠️ Collection '{self.collection_name}' does not exist, creating it...")
            self.create_collection()
            self.collection = Collection(self.collection_name)
            self.create_index()
            self.collection.load()
            print(f"✅ Collection '{self.collection_name}' has been created and loaded successfully.")

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",  # or any other index type you want
            metric_type="IP",  # or the appropriate metric type
            params={
                "M": 16,
                "efConstruction": 500,
            },  # adjust these parameters as needed
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        # Create a scalar index for the "doc_id" field to enable fast lookups by document ID.
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",  # or any other index type you want
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    @timeit
    def search(self, data, topk, partition=None):
        # Perform a vector search on the collection to find the top-k most similar documents.
        results = self.first_round_search(data, partition)
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_ids.add(results[r_id][r]["entity"]["doc_id"])

        scores = self.get_relevant_docs(data, doc_ids)

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            # Get a certain number of pages
            return scores[:topk]
        else:
            return scores

    @timeit
    def first_round_search(self, data, partition):
        search_params = {"metric_type": "IP", "params": {}}

        if FAST_MODE:
            RELEVANT_DOCS = 1
        else:
            RELEVANT_DOCS = 50
        if partition:
            results = self.client.search(
                self.collection_name,
                data,
                limit=int(RELEVANT_DOCS),
                output_fields=["vector", "seq_id", "doc_id"],
                search_params=search_params,
                partition_names=partition,
            )
        else:
            results = self.client.search(
                self.collection_name,
                data,
                limit=int(RELEVANT_DOCS),
                output_fields=["vector", "seq_id", "doc_id"],
                search_params=search_params,
            )
        return results

    @timeit
    def get_relevant_docs(self, data, doc_ids):
        print("Number of doc_ids:", len(doc_ids))
        print("Number of unique doc_ids:", len(set(doc_ids)))
        # Search for full-page books based on relevant patches.
        scores = []

        @timeit
        def rerank_single_doc(doc_id, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )

            # Take the doc field where seq_id==0 as the filepath
            filepath = ""
            for item in doc_colbert_vecs:
                if item.get("seq_id", 0) == 0:
                    filepath = item.get("doc", "")
                    break
            score = np.dot(data, doc_vecs.T).max(1).mean()
            return (score, doc_id, filepath, doc_vecs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in set(doc_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, filepath, doc_vecs = future.result()
                scores.append((score, doc_id, filepath, doc_vecs))
        return scores

    def insert(self, data, partition_name=None):
        # Insert ColBERT embeddings and metadata for a document into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [""] * seq_length
        docs[0] = data["filepath"]

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        if not partition_name:
            self.client.insert(
                self.collection_name,
                [
                    {
                        "vector": colbert_vecs[i],
                        "seq_id": seq_ids[i],
                        "doc_id": doc_ids[i],
                        "doc": docs[i],
                    }
                    for i in range(seq_length)
                ],
            )
        else:
            self.client.insert(
                self.collection_name,
                [
                    {
                        "vector": colbert_vecs[i],
                        "seq_id": seq_ids[i],
                        "doc_id": doc_ids[i],
                        "doc": docs[i],
                    }
                    for i in range(seq_length)
                ],
                partition_name=partition_name,
            )

    def insert_batch(self, data, partition_name=None):
        # data["colbert_vecs"]: numpy array (batch_size, seq_len, dim) 或 (batch_size, dim)
        # data["doc_id"]: list of length batch_size
        # data["filepath"]: list of length batch_size
        batch_size = len(data)
        entities_to_insert = []

        for i in range(batch_size):
            colbert_vecs = data["colbert_vecs"][i]
            if colbert_vecs.ndim == 1:  # Single vector case
                colbert_vecs = colbert_vecs.reshape(1, -1)
            seq_length = colbert_vecs.shape[0]
            doc_ids = [data["doc_id"][i]] * seq_length
            seq_ids = list(range(seq_length))
            docs = [""] * seq_length
            docs[0] = data["filepath"][i]

            entities = [
                {
                    "vector": colbert_vecs[j],
                    "seq_id": seq_ids[j],
                    "doc_id": doc_ids[j],
                    "doc": docs[j],
                }
                for j in range(seq_length)
            ]
            entities_to_insert.extend(entities)

        if partition_name:
            self.client.insert(
                self.collection_name, entities_to_insert, partition_name=partition_name
            )
        else:
            self.client.insert(self.collection_name, entities_to_insert)

    def exists_by_doc_id(self, doc_id: int) -> bool:
        results = self.collection.query(
            expr=f"doc_id == {doc_id}", output_fields=["doc_id"], limit=1
        )
        return len(results) > 0


