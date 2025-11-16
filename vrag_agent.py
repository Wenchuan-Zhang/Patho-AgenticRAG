import base64
import json
import re
import requests
import math
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw



from datetime import datetime

prompt_ins = """
Your task is **not** to directly answer the question, but to assist a downstream system by **retrieving and summarizing factual information** relevant to answering it.

You should focus on **describing the characteristics, criteria, or diagnostic features** associated with the possible answer—**not** on concluding what the answer is.

For example, if the question is whether an image shows invasive ductal carcinoma, you should **retrieve facts about how invasive ductal carcinoma typically presents**, especially in medical images or pathology reports.

Use reasoning steps enclosed in <think>...</think> to guide your process.

You may request external information using <search>...</search> or issue search queries using <search> query </search>. The user will provide relevant results. 
If a new image is retrieved, you may optionally crop it using <bbox>[x1, y1, x2, y2]</bbox>.

Base your reasoning on the image, the question, and any supporting data you collect. 
Use as many retrievals as necessary. Never rely solely on the initial image.

Once you have gathered enough information, provide a concise summary of the most **relevant evidence or diagnostic criteria** within <answer> and </answer> tags.

Do not state whether the diagnosis or conclusion is correct—just provide the features or findings that are relevant for making that decision.

Question: {question}
"""

first_turn = """
Your task is not to directly answer the question, but to assist a downstream system by retrieving and summarizing the most relevant information needed to answer it. 

You must conduct reasoning inside <think> and </think> each time new information is retrieved. 
Base your understanding on both the input image and the question, but do not simply describe or crop the image unless needed for clarity. 

Use <search> query </search> to issue a search query; the user will return the corresponding results. 

Once you've collected sufficient information, summarize the key facts or findings within <answer> and </answer>. 
Do not attempt to answer the question—just return a concise summary of the evidence relevant to answering it.

Question: {question}
"""


class VRAG:
    def __init__(
        self,
        base_url='http://localhost:8003/v1',  
        search_url="http://localhost:8001/search",
        generator=True,
        api_key="EMPTY",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.search_url = search_url

        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 3

        self.generator = generator

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = (
                int(image.width * resize_factor),
                int(image.height * resize_factor),
            )
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = (
                int(image.width * resize_factor),
                int(image.height * resize_factor),
            )
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{base64_string}"

        return image, base64_qwen

    def search(self, query, image=None, partition=None):
        """
            query: str or list[str]
            image: PIL.Image, path, base64, dict, or None
        partition: str or None
        """
        if isinstance(query, str):
            query = [query]
        data = {"queries": json.dumps(query)}
        # include partition if provided
        if partition is not None:
            data["partition"] = partition

        if image is not None:
            # support PIL.Image, path, base64, dict
            if isinstance(image, str):
                if image.startswith("data:image"):
                    img_base64 = image
                else:
                    with open(image, "rb") as f:
                        img_bytes = f.read()
                    img_base64 = (
                        "data:image;base64," + base64.b64encode(img_bytes).decode()
                    )
            elif isinstance(image, dict) and "bytes" in image:
                img_base64 = (
                    "data:image;base64," + base64.b64encode(image["bytes"]).decode()
                )
            elif isinstance(image, Image.Image):
                byte_stream = BytesIO()
                image.save(byte_stream, format="JPEG")
                img_base64 = (
                    "data:image;base64,"
                    + base64.b64encode(byte_stream.getvalue()).decode()
                )
            else:
                img_base64 = None
            if img_base64:
                data["images"] = json.dumps([img_base64])
        search_response = requests.post(self.search_url, json=data)
        search_results = search_response.json()
        # search_results: [[text topk], [image topk]]
        if image is not None and len(search_results) >= 2:
            text_topk = search_results[0]
            image_topk = search_results[1]
            text_files = {item["image_file"]: item for item in text_topk}
            image_files = {item["image_file"]: item for item in image_topk}
            intersection = set(text_files.keys()) & set(image_files.keys())
            if intersection:
                # Take the highest similarity in the intersection
                best_file = None
                best_score = -float("inf")
                for fname in intersection:
                    score = text_files[fname].get("score", 0)

                    if score > best_score:
                        best_score = score
                        best_file = fname
                
                # Save the final result
                final_result = [best_file]
                return final_result
            else:
                # Return one image from each (their respective top 1).
                result = []
                if text_topk:
                    result.append(text_topk[0]["image_file"])
                if image_topk:
                    result.append(image_topk[0]["image_file"])
                return result
        else:
            # Using only text queries
            # print(search_results)
            image_path_list = [result["image_file"] for result in search_results[0]]
            # Save the final result
            return image_path_list



    # def run(self, question):
    #     self.image_raw = []
    #     self.image_input = []
    #     self.image_path = []
    #     prompt = prompt_ins.format(question=question)
    #     messages = [dict(
    #         role="user",
    #         content=[
    #             {
    #                 "type": "text",
    #                 "text": prompt,
    #             }
    #         ]
    #     )]
    def run(self, question, image=None, partition=None):
        self.image_raw = []
        self.image_input = []
        self.image_path = []
        prompt = prompt_ins.format(question=question)
        first_message = first_turn.format(question=question)
        user_content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        first_user_content = [
            {
                "type": "text",
                "text": first_message,
            }
        ]
        if image is not None:
            # support str path or PIL.Image
            if isinstance(image, str):
                image = Image.open(image)
            image_input, img_base64 = self.process_image(image)
            user_content.append({"type": "image_url", "image_url": {"url": img_base64}})
            first_user_content.append(
                {"type": "image_url", "image_url": {"url": img_base64}}
            )
            # self.image_raw.append(image)
            # self.image_input.append(image_input)
        messages = [dict(role="user", content=user_content)]
        first_messages = [dict(role="user", content=first_user_content)]

        max_steps = self.max_steps
        while True:
            ## assistant
            if max_steps == self.max_steps:
                response = self.client.chat.completions.create(
                    model="VRAG-Agent",
                    messages=first_messages,
                    stream=False,
                )
            else:
                response = self.client.chat.completions.create(
                    model="VRAG-Agent", messages=messages, stream=False
                )
            response_content = response.choices[0].message.content
            messages.append(
                dict(
                    role="assistant",
                    content=[{"type": "text", "text": response_content}],
                )
            )
            ## think
            pattern = r"<think>(.*?)</think>"
            match = re.search(pattern, response_content, re.DOTALL)
            thought = match.group(1)
            if self.generator:
                yield "think", thought, match.group(0)
            ## opration
            pattern = r"<(search|answer|bbox)>(.*?)</\1>"
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                raw_content = match.group(0)
                content = match.group(
                    2
                ).strip()  # Return only the content inside the tags
                action = match.group(1)
            else:
                content = ""
                action = None

            ## whether end
            if action == "answer":
                return "answer", content, raw_content
            elif max_steps == 0:
                # return (
                #     "answer",
                #     "Sorry, I can not retrieval something about the question.",
                #     "",
                # )
                return ('', '')
            ## action
            # if action == "search":
            #     search_results = self.search(content, image, partition)
            #     while len(search_results) > 0:
            #         image_path = search_results.pop(0)
            #         # fix path - extract last two levels and append to textbook
            #         path_parts = image_path.split('/')
            #         # 获取最后两层：目录名和文件名
            #         last_two_levels = '/'.join(path_parts[-2:])  # 例如: "Skin_Dermatology/MCKEE'S PATHOLOGY of the Skin 4th edition2012_1218.jpg"
            #         # 构建新路径
            #         image_path = f"/11_data/gjr/RAG/patho-rag/VRAG/data/textbook/{last_two_levels}"
            #         if self.image_path.count(image_path) >= self.repeated_nums:
            #             continue
            #         else:
            #             self.image_path.append(image_path)
            #             break
            if action == "search":
                search_results = self.search(content, image, partition)
                for image_path in search_results:
                    # Skip already used paths
                    if self.image_path.count(image_path) < self.repeated_nums:
                        self.image_path.append(image_path)
                        break
                image_raw = Image.open(image_path)
                image_input, img_base64 = self.process_image(image_raw)
                user_content = [{"type": "image_url", "image_url": {"url": img_base64}}]
                self.image_raw.append(image_raw)
                self.image_input.append(image_input)
                if self.generator:
                    yield "search_image", self.image_input[-1], raw_content
            elif action == "bbox":
                bbox = json.loads(content)
                input_w, input_h = self.image_input[-1].size
                raw_w, raw_h = self.image_raw[-1].size
                crop_region_bbox = (
                    bbox[0] * raw_w / input_w,
                    bbox[1] * raw_h / input_h,
                    bbox[2] * raw_w / input_w,
                    bbox[3] * raw_h / input_h,
                )
                pad_size = 56
                crop_region_bbox = [
                    max(crop_region_bbox[0] - pad_size, 0),
                    max(crop_region_bbox[1] - pad_size, 0),
                    min(crop_region_bbox[2] + pad_size, raw_w),
                    min(crop_region_bbox[3] + pad_size, raw_h),
                ]
                crop_region = self.image_raw[-1].crop(crop_region_bbox)
                image_input, img_base64 = self.process_image(crop_region)
                user_content = [{"type": "image_url", "image_url": {"url": img_base64}}]
                self.image_raw.append(crop_region)
                self.image_input.append(image_input)

                if self.generator:
                    image_to_draw = self.image_input[-2].copy()
                    draw = ImageDraw.Draw(image_to_draw)
                    draw.rectangle(bbox, outline=(160, 32, 240), width=7)
                    yield "crop_image", self.image_input[-1], image_to_draw

            max_steps -= 1
            if max_steps == 0:
                user_content.append(
                    {
                        "type": "text",
                        "text": "please answer the question now with answer in <answer> ... </answer>",
                    }
                )
            messages.append(dict(role="user", content=user_content))

# if __name__ == "__main__":
#     agent = VRAG()
#     # generator = agent.run('What feature in the provided micrograph is indicative of chronic inflammation? /n A. Granuloma formation /n B. Multinucleated giant cells /n C. Neutrophilic infiltration /n D. Plasma cells with eccentrically placed nuclei', image='/11_data/gjr/PathGPT/example/images/241.jpg')
#     # generator = agent.run("How to differentiate ductal, papillary, and mucinous breast carcinoma histologically", "/11_data/gjr/RAG/patho-rag/1031753681417_.pic.jpg")
#     generator = agent.run("This image shows a ceruminous adenoma of the external ear canal. What is the key cytological feature of the cells lining the glandular structures?", "/11_data/gjr/RAG/patho-rag/1371753865532_.pic.jpg")
#     # generator = agent.run("This image shows a squamous papilloma of the oral cavity. Which of the following best describes its characteristic architecture?", "/11_data/gjr/RAG/patho-rag/1291753792650_.pic.jpg")
    
#     while True:
#         print(next(generator))
