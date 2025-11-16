import os
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:27890"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:27890"
from typing import List
from PIL import Image
from tqdm import tqdm
import torch
from pymilvus import MilvusClient
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor
from milvus import MilvusColQwen  
import hashlib


def generate_doc_id(filepath: str) -> int:
    return int(hashlib.md5(filepath.encode()).hexdigest()[:8], 16)

class ImageIngestion:
    def __init__(self, image_folder: str, milvus_qwen, model_name="vidore/colqwen2-v1.0", device_str="cuda:0", use_partition=False):
        self.image_folder = image_folder
        self.milvus_qwen = milvus_qwen
        self.model_name = model_name
        self.device = get_torch_device(device_str)
        self.use_partition = use_partition

        # Loading models and processors
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def _load_image_paths(self) -> List[str]:
        if not self.use_partition:
            return [
                os.path.join(self.image_folder, name)
                for name in os.listdir(self.image_folder)
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
            ]
        else:
            # Returns a list of (partition_name, image_path)
            image_paths = []
            for partition in os.listdir(self.image_folder):
                partition_path = os.path.join(self.image_folder, partition)
                if not os.path.isdir(partition_path):
                    continue
                for name in os.listdir(partition_path):
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        image_paths.append((partition, os.path.join(partition_path, name)))
            return image_paths

    def run(self):
        total = self.milvus_qwen.collection.num_entities
        print(f"Milvus {self.milvus_qwen.collection_name}Current total：{total} items.")

        if not self.use_partition:
            filepaths = self._load_image_paths()
            images = []
            paths_to_embed = []
            doc_ids = []

            for path in tqdm(filepaths, desc="🔍 Processing image paths"):
                doc_id = generate_doc_id(path)
                if self.milvus_qwen.exists_by_doc_id(doc_id):
                    print(f"⚠️ Already exists, skip: {path}")
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    paths_to_embed.append(path)
                    doc_ids.append(doc_id)
                except Exception as e:
                    print(f"❌ Failed to read: {path}, Error: {e}")

            if not images:
                print("✅ No new images need processing; all have been skipped.")
                return

            dataloader = DataLoader(
                dataset=ListDataset[str](images),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x),
            )

            num_inserted = 0
            for i, batch in enumerate(tqdm(dataloader, desc="🔍 Extract and insert into vector")):
                with torch.no_grad():
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    output = self.model(**batch)
                    vec = output[0].to("cpu")  # Because batch_size=1, we directly take the first one.
                    
                    data = {
                        "colbert_vecs": vec.float().numpy(),
                        "doc_id": doc_ids[i],
                        "filepath": paths_to_embed[i] if not self.use_partition else paths[i],
                    }
                    
                    if self.use_partition:
                        self.milvus_qwen.insert(
                            partition_name=partition,
                            data=data
                        )
                    else:
                        self.milvus_qwen.insert(data)
                    num_inserted += 1

            self.milvus_qwen.collection.flush()
            total = self.milvus_qwen.collection.num_entities
            msg = f"[{partition}] " if self.use_partition else ""
            print(f"✅ {msg}has inserted {num_inserted} vectors，Milvus {self.milvus_qwen.collection_name}current total：{total} vectors。")
        else:
            # Traverse by partition directory
            for partition in os.listdir(self.image_folder):
                partition_path = os.path.join(self.image_folder, partition)
                if not os.path.isdir(partition_path):
                    continue

                # Processing images from a single partition
                images = []
                paths = []
                doc_ids = []

                # Load images from the current partition
                for name in tqdm(os.listdir(partition_path), desc=f"🔍 Process images in partition {partition}"):
                    if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        continue
                    
                    path = os.path.join(partition_path, name)
                    doc_id = generate_doc_id(path)
                    if self.milvus_qwen.exists_by_doc_id(doc_id):
                        print(f"⚠️ Already exists, skip: {path}")
                        continue
                    
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        paths.append(path)
                        doc_ids.append(doc_id)
                    except Exception as e:
                        print(f"❌ Failed to read: {path}, Error: {e}")

                if not images:
                    print(f"⚠️ No new images need to be processed in partition {partition}")
                    continue

                # Create partition
                try:
                    self.milvus_qwen.client.create_partition(
                        collection_name=self.milvus_qwen.collection_name,
                        partition_name=partition
                    )
                except Exception as e:
                    print(f"⚠️ Failed to create partition {partition} or it already exists: {e}")

                dataloader = DataLoader(
                    dataset=ListDataset[str](images),
                    batch_size=1,
                    shuffle=False,
                    collate_fn=lambda x: self.processor.process_images(x),
                    num_workers=16
                )

                num_inserted = 0
                data = []
                for i, batch in enumerate(tqdm(dataloader, desc=f"🔍 [{partition}] Extract and insert into vector")):
                    with torch.no_grad():
                        batch = {k: v.to(self.model.device) for k, v in batch.items()}
                        output = self.model(**batch)
                        vec = output[0].to("cpu")  # Because batch_size=1, we directly take the first one.
                        
                        data = {
                            "colbert_vecs": vec.float().numpy(),
                            "doc_id": doc_ids[i],
                            "filepath": paths[i],
                        }
                        self.milvus_qwen.insert(
                            partition_name=partition,
                            data=data
                        )
                        num_inserted += 1
                self.milvus_qwen.collection.flush()
                total = self.milvus_qwen.collection.num_entities
                
                print(f"✅ [{partition}] has inserted {num_inserted} vectors. The current total number of Milvus {self.milvus_qwen.collection_name} vectors is {total}.")

if __name__ == "__main__":
    image_folder = "./pdf_datasets"
    milvus_client = MilvusClient(host="127.0.0.1", port=19530)
    
    milvus_qwen = MilvusColQwen(milvus_client, "pdf_datasets", dim=128)

    # Add the `use_partition` parameter
    ingestion = ImageIngestion(image_folder=image_folder, milvus_qwen=milvus_qwen, use_partition=True)
    ingestion.run()