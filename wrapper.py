# file：simple_wrapper.py
import requests

class SimpleOpenAIWrapper:
    def __init__(self, model, base_url, api_key):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    class Chat:
        def __init__(self, parent):
            self.parent = parent

        def completions(self):
            return self

        def create(self, model, messages, timeout=None):
            url = f"{self.parent.base_url}/chat/completions"
            payload = {"model": model, "messages": messages}
            headers = {"Authorization": f"Bearer {self.parent.api_key}"}
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

    @property
    def chat(self):
        return self.Chat(self)
