import os

import time

from ollama import chat
from ollama import ChatResponse

from pydantic import BaseModel

# step 1: define the response in structured format
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# step 2: call the model
completion = chat(

        model='llama3.2', 
        messages=[
            {"role": "system", "content": "Extract the event details" },
            {
                'role': 'user',
                'content': "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        format=CalendarEvent.model_json_schema(),
    )

# step 3: parse the response
event = CalendarEvent.model_validate_json(completion.message.content)
print(f"Event:{event.name}\nDate:{event.date}\nParticipants:{event.participants}")