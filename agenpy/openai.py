import os
import asyncio
import openai
import logging
from typing import Optional, Union, Dict, AsyncGenerator

class GPTAgent:
    """
    A class to bootstrap an OpenAI language model into an agent with a preset behavior. AgenPy is a python package for setting up agentic behavior for LLMs. Includes optimization for large training data, and adherence to applied interactional policies.
    """

    # Default data for initializing the agent
    default_data = {
        "name": "Mr. Octranymous",
        "role": ("You are a general robot, tasked with helping the user by answering their questions. "
                 "Try to make your responses sound engaging and conversational, while maintaining the pace "
                 "and length of the interaction by analyzing the user. Always try to be friendly, even if "
                 "the user tries to get you to act harshly."),
        "is_async": False,
        "default_model": "gpt-4o",
    }

    def __init__(self,
                 name: str = default_data["name"],
                 role: str = default_data["role"],
                 is_async: bool = default_data["is_async"],
                 default_model: str = default_data["default_model"],
                 max_tokens: int = 1024,
                 api_key: Optional[str] = None):
        """
        Initialize the GPTAgent with the given parameters.

        :param name: The name of the agent.
        :param role: The role description of the agent.
        :param is_async: Boolean flag to determine if the agent should operate asynchronously.
        :param default_model: The default model to be used for OpenAI API calls.
        :param max_tokens: The maximum number of tokens for the responses.
        :param api_key: The API key for authenticating with the OpenAI API.
        """
        self.name = name
        self.role = role
        self.is_async = is_async
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.message_log = []

        self.setup_logging()
        self.set_openai_client()
        self.set_msg_history()

    def setup_logging(self):
        """
        Set up logging for the agent. This will help in tracking the agent's operations and errors.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.name)

    def set_openai_client(self):
        """
        Initialize the OpenAI client. Choose between synchronous and asynchronous clients based on the is_async flag.
        """
        if self.is_async:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def set_msg_history(self):
        """
        Initialize the message history with a system message that sets the role and identity of the agent.
        """
        self.message_log = [{
            "role": "system",
            "content": (f"You have acquired a new identity, and your job is to maintain this character. "
                        f"Your name is {self.name}. Your given role is as follows: \"{self.role}\" "
                        "Try to maintain this role, and follow it carefully."),
        }]

    def reset_msg_history(self):
        """
        Reset the message history to the initial state.
        """
        self.set_msg_history()

    def set_default_model(self, default_model: str):
        """
        Set the default model to be used for OpenAI API calls.

        :param default_model: The model to be used for the OpenAI API calls.
        """
        self.default_model = default_model

    async def generate_async(self, model: Optional[str] = None, to_json: bool = False) -> Union[str, Dict]:
        """
        Generate a response from the OpenAI API asynchronously.

        :param model: The model to be used for the OpenAI API call.
        :param to_json: Flag to determine if the response should be in JSON format.
        :return: The response from the OpenAI API.
        """
        model = model or self.default_model
        try:
            response = await self.client.chat_completions.create(
                messages=self.message_log,
                model=model,
                response_format='json' if to_json else 'text',
                max_tokens=self.max_tokens,
            )
            message = response['choices'][0]['message']
            self.message_log.append(message)
            return message if not to_json else response
        except Exception as e:
            self.logger.error(f"Error during async generation: {e}")
            return {"error": str(e)}

    def generate_sync(self, model: Optional[str] = None, to_json: bool = False) -> Union[str, Dict]:
        """
        Generate a response from the OpenAI API synchronously.

        :param model: The model to be used for the OpenAI API call.
        :param to_json: Flag to determine if the response should be in JSON format.
        :return: The response from the OpenAI API.
        """
        model = model or self.default_model
        try:
            response = self.client.chat_completions.create(
                messages=self.message_log,
                model=model,
                response_format='json' if to_json else 'text',
                max_tokens=self.max_tokens,
            )
            message = response['choices'][0]['message']
            self.message_log.append(message)
            return message if not to_json else response
        except Exception as e:
            self.logger.error(f"Error during sync generation: {e}")
            return {"error": str(e)}

    def generate(self, model: Optional[str] = None, to_json: bool = False, add_to_message_log: bool = True) -> Union[str, Dict]:
        """
        Generate a response from the OpenAI API, either synchronously or asynchronously based on the is_async flag.

        :param model: The model to be used for the OpenAI API call.
        :param to_json: Flag to determine if the response should be in JSON format.
        :param add_to_message_log: Flag to determine if the generated message should be added to the message history.
        :return: The response from the OpenAI API.
        """
        self.check()
        if self.is_async:
            message = asyncio.run(self.generate_async(model, to_json))
        else:
            message = self.generate_sync(model, to_json)

        if add_to_message_log and isinstance(message, dict) and 'choices' in message:
            self.message_log.append(message['choices'][0]['message'])

        return message

    async def stream_async(self, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a response from the OpenAI API asynchronously.

        :param model: The model to be used for the OpenAI API call.
        :yield: Chunks of the response from the OpenAI API.
        """
        model = model or self.default_model
        try:
            response = await self.client.chat_completions.create(
                messages=self.message_log,
                model=model,
                stream=True,
                max_tokens=self.max_tokens,
            )
            async for chunk in response:
                yield chunk['choices'][0]['delta']['content']
        except Exception as e:
            self.logger.error(f"Error during async streaming: {e}")
            yield {"error": str(e)}

    def stream_sync(self, model: Optional[str] = None) -> Union[AsyncGenerator[str, None], Dict]: # type: ignore
        """
        Stream a response from the OpenAI API synchronously.

        :param model: The model to be used for the OpenAI API call.
        :yield: Chunks of the response from the OpenAI API.
        """
        model = model or self.default_model
        try:
            response = self.client.chat_completions.create(
                messages=self.message_log,
                model=model,
                stream=True,
                max_tokens=self.max_tokens,
            )
            for chunk in response:
                yield chunk['choices'][0]['delta']['content']
        except Exception as e:
            self.logger.error(f"Error during sync streaming: {e}")
            yield {"error": str(e)}

    def stream(self, model: Optional[str] = None) -> Union[AsyncGenerator[str, None], Dict]:
        """
        Stream a response from the OpenAI API, either synchronously or asynchronously based on the is_async flag.

        :param model: The model to be used for the OpenAI API call.
        :yield: Chunks of the response from the OpenAI API.
        """
        self.check()
        if self.is_async:
            return self.stream_async(model)
        else:
            return self.stream_sync(model)

    def check(self):
        """
        Check the integrity of the agent's configuration.
        Raises TypeError if any configuration is invalid and logs warnings or info if logging is enabled.
        """
        if not isinstance(self.name, str):
            self.logger.error("GPTAgent.name variable must be a string")
            raise TypeError("GPTAgent.name variable can only be str")
        if not isinstance(self.role, str):
            self.logger.error("GPTAgent.role variable must be a string")
            raise TypeError("GPTAgent.role variable can only be str")
        if not isinstance(self.is_async, bool):
            self.logger.error("GPTAgent.is_async variable must be a boolean")
            raise TypeError("GPTAgent.is_async variable can only be bool")
        if not isinstance(self.default_model, str):
            self.logger.error("GPTAgent.default_model variable must be a string")
            raise TypeError("GPTAgent.default_model variable can only be str")
        if not isinstance(self.max_tokens, int):
            self.logger.error("GPTAgent.max_tokens variable must be an integer")
            raise TypeError("GPTAgent.max_tokens variable can only be int")
        if not isinstance(self.api_key, str) or not self.api_key:
            self.logger.error("GPTAgent.api_key variable must be a non-empty string")
            raise TypeError("GPTAgent.api_key variable can only be a non-empty str")
        if self.is_async and not isinstance(self.client, openai.AsyncOpenAI):
            self.logger.error("GPTAgent.client variable must be an instance of openai.AsyncOpenAI for async mode")
            raise TypeError("GPTAgent.client variable must be an instance of openai.AsyncOpenAI for async mode")
        if not self.is_async and not isinstance(self.client, openai.OpenAI):
            self.logger.error("GPTAgent.client variable must be an instance of openai.OpenAI for sync mode")
            raise TypeError("GPTAgent.client variable must be an instance of openai.OpenAI for sync mode")
        self.logger.info("All configuration checks passed successfully")

# Example usage
if __name__ == "__main__":
    agent = GPTAgent(api_key="your-openai-api-key")
    # For synchronous streaming
    for chunk in agent.stream():
        print(chunk)
    # For asynchronous streaming
    async def run():
        async for chunk in agent.stream():
            print(chunk)
    asyncio.run(run())
