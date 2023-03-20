import os

import requests

from evals.base import ModelSpec
from evals.prompt.base import OpenAICreatePrompt

from .base import _ModelRunner


class LlamaRunner(_ModelRunner):
    def completion(self, prompt: OpenAICreatePrompt, **kwargs):
        # NOTE: run evals against LLAMA API server, you can get the LLAMA API server from https://github.com/open-evals/pyllama
        r = requests.post(f"{os.environ['LLAMA_SERVER']}/prompt", json={"prompts": [prompt], "temperature": 0, "top_p": 0.95})
        
        result = r.json()
        prompt_list = prompt.split("\n")
        result_list = [sentence for sentence in result["results"][0].split("\n") if sentence != ""]
        ans = ""
        for i, prompt in enumerate(prompt_list):
            if prompt != result_list[i]:                
                ans = result_list[i].replace(prompt, "")
                break
        ans = ans.split(":")[-1]
        return {"choices": [{"text": "".join(ans.strip())}]}

    @classmethod
    def resolve(cls, name: str) -> ModelSpec:
        if name.startswith("llama"):
            return ModelSpec(runner="llama", name=name, model=name, is_chat=False, n_ctx=2048)
        raise ValueError(f"Model {name} not found")
