import time

from ollama import chat
from ollama import ChatResponse

start = time.time()

completion = chat(

        model='llama3.2', 
        messages=[
            {"role": "system", "content": "You are a helpful assistant." },
            {
                'role': 'user',
                'content': "Write a limerick about python programming",
            },
        ],
    )

response = completion.message.content
print(response)