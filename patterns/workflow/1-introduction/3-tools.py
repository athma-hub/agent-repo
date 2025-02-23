import os

import time

from ollama import chat
from ollama import ChatResponse

from pydantic import BaseModel, Field

import requests
import pprint
import json

# Building blocks for building workflows, and agents - Building block: The augmented LLM - Tools (Call/Response) is one in the list, 
# the other 2 being Retreival (Query/Results) and Memory (Read/Write)

# we are going to use a weather API. This API we can model that as a tool which is available to the AI and then based on the context, the AI
# can look at the availble tools, then dependending on the user question the AI is going to decide if it wants to use the tool yes/no

# the AI will actually not call the tool for you, that is not how it works, the AI will provide only the parameters that are expected by the 
# tool (function) that has we want to be called. We have to actually put in the parameters inside the function logic and get what we want as o/p

# ------------------------------------------------------------------------------------------------------------
# Define the tool (function) that we want to call
# There is no AI here, it is just a publicly avaiable API that will be used to call and get information
# You can use your own API or any other API it does not matter and it is not connected with the llm and AI
# ------------------------------------------------------------------------------------------------------------
def get_weather(latitude, longitude):
    """This is a publicly available API that returns the weather for a given location"""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

# Below is an example of what an actual tool use looks like

# ------------------------------------------------------------------------------------------------------------
# step 1: call model with the get_weather tool defined
# ------------------------------------------------------------------------------------------------------------
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a provided coordinates in Celcius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    },
}]

system_prompt = "You are a helpful weather assistant."

messages=[
    {"role": "system", "content": system_prompt },
    {"role": "user", "content": "What's the weather like in Chennai today?"},
]

# note now we have to specify the tools to the model as well to help it in deciding the tool to be called
completion = chat(
    model="llama3.2",
    messages=messages,
    tools=tools,
)

# ------------------------------------------------------------------------------------------------------------
# step 2: model decide to call function(s)
# ------------------------------------------------------------------------------------------------------------
pprint.pprint(completion.model_dump())


# ------------------------------------------------------------------------------------------------------------
# step 3: Execute the get_weather function
# Please note that we call the function ourself, the AI does not call the function it only passes back the 
# functon name to call and the arguments. We use that call the function and also build the memory by addind
# to the messages object the message got back from the llm and also the result for from the tool call we made
# to keep building on the memiory and contect
# ------------------------------------------------------------------------------------------------------------
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

for tool_call in completion.message.tool_calls:
    name = tool_call.function.name # name of the function got back from the llm based on the available tools and user question
    # args = json.loads(tool_call.function.arguments) # arguments is always in the form of JSON - OpenAI
    args = tool_call.function.arguments # in case of llama
    
    # Memory: Following the message syntax add the new output to the messages list for context
    messages.append(completion.message)

    print(name)
    print(args)
    pprint.pprint(messages)

    result = call_function(name, args)
    print(result)

    # messages.append({"role": "tool", "tool_Call_id": tool_call.id, "content": str(result)}) # in case of OpenAI you can also add the tool_id
    messages.append({"role": "tool", "content": str(result)}) # in case of llama
    pprint.pprint(messages)

# ------------------------------------------------------------------------------------------------------------
# step 4: Supply the result and call the model again
# Now we have the full context that we need to answer the intial question the user. We will use structure output
# again from the final answer from the llm
# ------------------------------------------------------------------------------------------------------------
class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in celcius for the given coordinates")
    response: str = Field(description="A natural language response to the user's question")

# now we will be calling the model again, but this time not we will not be sending just the original message but
# we will be sending the entire context available - the orginal message, tools available, along with the tool call 
# information obtained from the first call to the llm and the result obtained from executing the tool call to the llm
# and will be asking for the response from the llm in the structure output format we have defined namely WeatherResponse
completion_final = chat(
    model="llama3.2",
    messages=messages,
    tools=tools,
    format=WeatherResponse.model_json_schema(),
)

# ------------------------------------------------------------------------------------------------------------
# step 5: Checkthe model response
# ------------------------------------------------------------------------------------------------------------
final_response = WeatherResponse.model_validate_json(completion_final.message.content)
print(f"Response:{final_response.response}\nTemperature:{final_response.temperature}\n")