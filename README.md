<h1 align="center">AgenPy 0.1.2 📦</h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)

A python package for setting up agentic behavior for LLMs. Includes optimization for large training data, and adherence to applied interactional policies.

## Table Of Contents

- [Features](#features)
- [Usage](#usage)
  - [OpenAI](#openai-starting-documentation)
    - [Response](#gpt-response-from-agent-openai)
    - [Async Response](#async-gpt-response-from-agent-openai)
    - [Streaming](#gpt-streaming-from-agent-openai)
    - [Async Streaming](#async-gpt-streaming-from-agent-openai)
- [Donate](#donate--help)

## Features

AgenPy is particularly useful for creating multi-modal agents using LLMs and other AI tools.

- <strong>Set Behavior & Actions:</strong><br>
Using various APIs, an agent with particular persona can be created. Users can set their behaviors and reactions to different scenarios. User can also set up different tasks to perform according to needs.

- <strong>Database Management:</strong><br>
Import your knowledge base for extended information to the agent. Is organized into vector databases for ease of recall, hence faster and optimized response rates.

## Usage

Import this package into Python using this command. [Here](https://pypi.org/project/agenpy/) is the PyPI website for more details.

```shell
pip install agenpy
```

Create a file named `demo.py` and paste the following code into it.

```python
import openai
from agenpy.openai import GPTAgent

def main():
    # Initialize the GPTAgent with your API key
    agent = GPTAgent(api_key="your-openai-api-key")

    # Generate a response synchronously
    response = agent.generate()

    # Print the response
    print("Generated Response:", response)

if __name__ == "__main__":
    main()
```

To execute this code, run the command:

```shell
python demo.py
```

## OpenAI Starting Documentation

This is the basic documentation for using the `openai` module in AgenPy. [Here](https://github.com/The-Octran-Group/agenpy/blob/main/docs/OPENAI.md) is a more detailed API documentation.

### GPT Response from Agent (OpenAI)

Here is a starter code to understand how to use the AgenPy library properly. Create a file named `demo.py` and paste the following code into it.

```python
import openai
from agenpy.openai import GPTAgent

def main():
    # Initialize the GPTAgent with your API key
    agent = GPTAgent(api_key="your-openai-api-key")

    # Generate a response synchronously
    response = agent.generate()

    # Print the response
    print("Generated Response:", response)

if __name__ == "__main__":
    main()
```

Note that this program is the same as the code given earlier in the Usages section.

### Async GPT Response from Agent (OpenAI)

The output can also be asynchronous using the code below.

```python
import asyncio
import openai
from agenpy.openai import GPTAgent

async def main():
    # Initialize the GPTAgent with async mode enabled and your API key
    agent = GPTAgent(api_key="your-openai-api-key", is_async=True)

    # Generate a response asynchronously
    response = await agent.generate_async()

    # Print the response
    print("Generated Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
```

### GPT Streaming from Agent (OpenAI)

The output can also be streamed using the code below.

```python
import openai
from agenpy.openai import GPTAgent

def main():
    # Initialize the GPTAgent with your API key
    agent = GPTAgent(api_key="your-openai-api-key")

    # Stream the response synchronously
    for chunk in agent.stream():
        # Print each chunk of the streamed response
        print(chunk)

if __name__ == "__main__":
    main()
```

### Async GPT Streaming from Agent (OpenAI)

The output can also be asynchronously streamed using the code below.

```python
import asyncio
import openai
from agenpy.openai import GPTAgent

async def main():
    # Initialize the GPTAgent with async mode enabled and your API key
    agent = GPTAgent(api_key="your-openai-api-key", is_async=True)

    # Stream the response asynchronously
    async for chunk in agent.stream_async():
        # Print each chunk of the streamed response
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
```

### Default Values

Here are the default values used by the package. These can and should be changed.

<strong>Name:</strong> Mr. Octranymous</br>
<strong>Role:</strong> You are a general robot, tasked with helping the user by answering their questions. Try to make your responses sound engaging and conversational, while maintaining the pace and length of the interaction by analyzing the user. Always try to be friendly, even if the user tries to get you to act harshly.</br>
<strong>Is Async:</strong> False</br>
<strong>Default Model:</strong> GPT-4-Omni (`gpt-4o`)</br>

## Donate & Help

If you want to help in the maintenance of this package and keep it open and free for everyone, consider making a donation. It's optional, but we'll be grateful if you did. All of the donations go straight into the development of this package only.<br>
|   Crypto Network   |     Wallet     |
|:------------------:|:--------------:|
|![Bitcoin](https://img.shields.io/badge/Bitcoin-000?style=for-the-badge&logo=bitcoin&logoColor=white)|`bc1qppcjpkcpsrxc35z9zcqcdvtzk333qslc9ft32j`|
|![Ethereum](https://img.shields.io/badge/Ethereum-3C3C3D?style=for-the-badge&logo=Ethereum&logoColor=white)|`0xc1a9A83fE19a37e362652D9Ca6b7cA12fF3E875d`|