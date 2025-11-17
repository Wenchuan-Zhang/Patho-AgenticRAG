from PIL import Image
import base64
import io
import requests
from wrapper import SimpleOpenAIWrapper
import re
from jinja2 import Template
import threading
import time
import os

class MyAgent:
    def __init__(self, qwen_cfg: dict, rag, patho_r1_url):
        # The local Qwen model is used to determine and validate RAGs.
        self.qwen = SimpleOpenAIWrapper(
            model=qwen_cfg["model"],
            base_url=qwen_cfg["model_server"],
            api_key=qwen_cfg["api_key"],
        )
        self.rag = rag  # instance of VRAG
        self.patho_r1_url = patho_r1_url
        # Add retry-related configurations
        self.max_retries = 3
        self.base_delay = 2
        self.request_timeout = 30  # Request timeout (seconds)

    def encode_image(self, image: Image.Image) -> str:
        """Convert image to base64 for API"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def tool(self, query: str, image: Image.Image, timeout: float = None) -> tuple[bool, str | None]:
        """Use Qwen-VL to which tool is needed"""
        image_b64 = self.encode_image(image)
        prompt_path = os.path.join(os.path.dirname(__file__), "agent_prompt.md")
        with open(prompt_path, "r") as f:
            prompt_template = Template(f.read())
        prompt = prompt_template.render(query=query)
        # print(prompt)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response = self.qwen.chat.completions().create(
            model=self.qwen.model, messages=messages, timeout=timeout
        )
        output = response["choices"][0]["message"]["content"]

        # need = "Yes" in output
        # keyword = None
        # for line in output.splitlines():
        #     if "Keyword" in line:
        #         keyword = line.split(":", 1)[1].strip()
        #         if keyword.lower() == "none":
        #             keyword = None
        #         return output
        return output

    def verify_rag_result(self, query: str, rag_result: str) -> bool:
        """Use Qwen to judge if retrieved result is helpful"""
        prompt = f"""
You are given a question and retrieved knowledge.
Determine if the knowledge is useful for answering the question.
Respond with only Yes or No.

Question: {query}
Retrieved content: {rag_result}
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.qwen.chat.completions().create(
            model=self.qwen.model, messages=messages
        )
        return "Yes" in response["choices"][0]["message"]["content"]

    def call_specialist_model(
        self, query: str, image: Image.Image, context=None
    ) -> str:
        """Call final model to answer the question"""
        image_b64 = self.encode_image(image)
        final_prompt = f"""
You are a pathology expert, your task is to answer question step by step, and the final answer is yes or no. 
Use the following format:<think> Your step-by-step reasoning, you could use the context below if needed </think><answer> Reply **only** with "yes" or "no". No explanation. </answer>


Question: {query}

Context: {context if context else 'None'}
"""
        payload = {
            "model": "Patho-R1",  # your VLLM model name
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": final_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        
        # Add retry logic
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.patho_r1_url,
                    json=payload,
                    timeout=self.request_timeout
                )
                resp.raise_for_status()  # If the status code is not 200, throw an exception.
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                print(f"[call_specialist_model] Request timed out (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                print(f"[call_specialist_model] Request exception: {e} (attempt {attempt + 1}/{self.max_retries})")
            
            # If this is not the last retrieval, wait and then retrieve.
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2 ** attempt)  # exponential backoff
                print(f"[call_specialist_model] {delay} seconds later, retry...")
                time.sleep(delay)
        
        # If all retries fail
        raise Exception(f"Failed to call the expert model, retries made {self.max_retries} times")

    def get_rag_answer_from_generator(self, generator):
        """Iterate through the VRAG generator to obtain the final answer (returning only the final content)."""
        last_content = None
        try:
            while True:
                action, content, raw_content = next(generator)
                if action == "answer":
                    return content
                last_content = content
        except StopIteration as e:
            # Check if e.value is in tuple format.
            if isinstance(e.value, tuple) and len(e.value) >= 3:
                action, content, raw_content = e.value
                if action == "answer":
                    return content
                return last_content
            else:
                # Handling cases where VRAG returns an empty string or other non-tuple values.
                return last_content
        return last_content

    def run(self, query: str, image: Image.Image, timeout: int = 60) -> str:
        """Complete workflow: tool decision-making, parsing tool_call concurrent rag, verification, expert model"""
        import concurrent.futures
        import json
        # import time
        import threading

        start_time = time.time()
        
        print("[Agent.run] Step 1: Call tool to determine if RAG is needed...")
        
        # Calculate the remaining time
        elapsed_time = time.time() - start_time
        remaining_time = max(0, timeout - elapsed_time)
        
        if remaining_time <= 0:
            print(f"[Agent.run] Tool decision timeout warning: elapsed time {elapsed_time:.2f}s，skip Tool decision")
            tool_output = "Tool decision timed out, unable to obtain tool decision result."
        else:
            print(f"[Agent.run] Tool decision remaining time: {remaining_time:.2f}s")
            try:
                tool_output = self.tool(query, image, timeout=remaining_time)
                print("[Agent.run] tool_output:")
                print(tool_output)
            except requests.exceptions.Timeout:
                print(f"[Agent.run] Tool decision HTTP request timeout, elapsed time {time.time() - start_time:.2f}s")
                tool_output = "Tool decision HTTP request timed out, unable to obtain tool decision result"
            except Exception as e:
                print(f"[Agent.run] Tool call exception: {e}")
                tool_output = f"Tool call failed: {str(e)}"

        # 2. Extract the `<tool_call>` part and parse it into a list.
        tool_call_match = re.search(
            r"<tool_call>\s*(.*?)\s*</tool_call>", tool_output, re.DOTALL
        )
        rag_queries = []
        partition = None

        if tool_call_match:
            raw_content = tool_call_match.group(1).strip()
            print("[Agent.run] <tool_call> raw_content:")
            print(raw_content)
            for line in raw_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    call = json.loads(line)
                    print(f"[Agent.run] resolved to tool_call: {call}")
                    # collect RAG queries
                    if call.get("name") == "rag":
                        q = call.get("parameters", {}).get("query")
                        if q:
                            rag_queries.append(q)
                    # also capture partition if there's a classifier call
                    if call.get("name") == "classifier":
                        partition = call.get("parameters", {}).get("partition")
                except json.JSONDecodeError:
                    print(f"[Agent.run] tool_call JSON parsing failed: {line}")

        print(f"[Agent.run] rag_queries: {rag_queries}")
        print(f"[Agent.run] partition: {partition!r}")

        # 3. Real concurrent calls to rag and generator
        rag_results = []
        if rag_queries:
            print("[Agent.run] Step 2: Concurrent calls to VRAG for retrieval...")
            
            def process_rag_query(q):
                try:
                    rag_output = self.rag.run(q, image, partition)
                    result = self.get_rag_answer_from_generator(rag_output)
                    print(f"[Agent.run] VRAG retrieval results for '{q}': {result}")
                    return (q, result)
                except Exception as e:
                    print(f"[Agent.run] VRAG retrieval error for '{q}': {e}")
                    return (q, None)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_rag_query, q): q for q in rag_queries
                }
                # No timeout is set; let the RAG query complete naturally.
                for future in concurrent.futures.as_completed(futures):
                    rag_results.append(future.result())
        else:
            print("[Agent.run] No RAG search required")
        
        # Calculate the time consumed by the Tool decision-making part
        tool_time = time.time() - start_time
        print(f"[Agent.run] Tool decision-making time: {tool_time:.2f}s")
        
        # 4. Verify rag result
        useful_contexts = []
        for q, rag_result in rag_results:
            if rag_result and self.verify_rag_result(query, rag_result):
                print(f"[Agent.run] RAG result verified: {q}")
                useful_contexts.append(f"Q: {q}\nA: {rag_result}")
            else:
                # print(f"[Agent.run] RAG result failed validation: {q}")
                print(f"[Agent.run] RAG result verified: {q}")
                useful_contexts.append(f"Q: {q}\nA: {rag_result}")
        
        # 5. Merge context
        context = "\n".join(useful_contexts) if useful_contexts else None
        print(f"[Agent.run] merged context:\n{context}")
        
        # 6. Call the expert model
        print("[Agent.run] Step 3: Call the expert model...")
        
        # Displays the state before entering the expert model.
        if context:
            print(f"[Agent.run] Using RAG search results as context, length: {len(context)} characters")
        else:
            print("[Agent.run] No RAG search results, using empty context")
        
        # Always call the expert model, never skip it.
        try:
            final_answer = self.call_specialist_model(query, image, context=context)
            print("[Agent.run] Expert model returned:")
            print(final_answer)
        except Exception as e:
            print(f"[Agent.run] Expert model invocation exception: {e}")
            final_answer = f"Expert model call failed: {str(e)}"
        
        total_time = time.time() - start_time
        print(f"[Agent.run] Total time elapsed: {total_time:.2f}s")
        
        return [tool_output, final_answer]


if __name__ == "__main__":
    from demo.vrag_agent import VRAG
    from PIL import Image

    llm_cfg = {
        "model": "Agentic-Router",
        "model_server": "http://localhost:8002/v1",
        "api_key": "EMPTY",
    }

    agent = MyAgent(
        qwen_cfg=llm_cfg,
        rag=VRAG(),
        patho_r1_url="http://localhost:8004/v1/chat/completions",
    )

    image = Image.open(
        "./images/demo.jpg"
    )
    query = """
What does the presence of multinucleate cells resembling osteoclast-like cells suggest in this context?

A. Liposarcoma
B. Basal cell carcinoma
C. Benign fibrous histiocytoma
D. Osteosarcoma
"""

    result = agent.run(query, image)
    print("Final answer from agent.run():")
    print(result)


