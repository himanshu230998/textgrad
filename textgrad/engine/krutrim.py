
import os
import platformdirs
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from .base import EngineLM, CachedEngine

url = "https://cloud.olakrutrim.com/v1/chat/completions"

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('KRUTRIM_API_KEY')}"
}

class Krutrim(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="Krutrim",
        system_prompt=SYSTEM_PROMPT,
        temperature=0,
        top_p = 0.8
    ):

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_krutim_{model_string}.db")
        super().__init__(cache_path=cache_path)
        if os.getenv("KRUTRIM_API_KEY") is None:
            raise ValueError("Please set the KRUTRIM_API_KEY environment variable if you'd like to use KRUTRIM_API_KEY models.")
        
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        assert isinstance(self.system_prompt, str)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.8
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        data = {
            "model": "Meta-Llama-3.1-8B-Kumbh",
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt_arg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        # Make the POST request
        response = requests.post(url, headers=headers, json=data)

        # Print the response
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error:", response.status_code, response.text)
        return response.json()['choices'][0]['message']['content']
