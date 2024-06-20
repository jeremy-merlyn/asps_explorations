import vertexai
from vertexai.language_models import TextGenerationModel

from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerationConfig

import requests

from openai import OpenAI
import agents_and_prompts


class MistralModel:
    def __init__(self, name):
        self.name = name
        print("Using %s model..." % name)

    def get_response(self, prompt, command):

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        history = [
            {"role": "system", "content": prompt},
        ]

        history.append({"role": "user", "content": "QUERY: " + command})
        history.append({"role": "assistant", "content": "CATEGORY NUMBER: "})
        
                        
        completion = client.chat.completions.create(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            messages=history,
            temperature=0.0,
            stop=["\n", "<|end|>", ".", " "],
            stream=False,
        )
        return completion.choices[0].message.content
    
    def get_response_tgi(self, prompt, command):

        PROMPT_OUTRO = "\n\n[INST] QUERY: %s [/INST] CATEGORY NUMBER: " % command

        url= "http://localhost:8080/generate"

        prompt = prompt + PROMPT_OUTRO
        data = {"inputs": prompt,
        "parameters": { 
        "stop": ["</s>",  " "],
        "details": False,
        "do_sample": False,
        "max_new_tokens": 1,
        "temperature": 0.01
            }
        }
    
        response = requests.post(url, json=data)
        print(response.json())
        return response.json()['generated_text']

class PhiModel:
    def __init__(self, name):
        self.name = name
        print("Using %s model..." % name)


    def get_response(self, prompt, command):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        history = [
            {"role": "system", "content": prompt},
        ]

        history.append({"role": "user", "content": "QUERY: " + command})
        history.append({"role": "assistant", "content": "CATEGORY NUMBER: "})
        
                        
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            messages=history,
            temperature=0.0,
            stop=["\n", "<|end|>", "."],
            stream=False,
        )
        return completion.choices[0].message.content


    

