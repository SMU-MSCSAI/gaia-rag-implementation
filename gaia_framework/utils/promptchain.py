import json
import re
from typing import List, Dict, Callable, Any, Union
from functools import lru_cache
from jinja2 import Template


class CustomPromptChain:
    """
    Class for dynamic prompt chaining with context, output back-references, and conversation history.
    """
    
    @staticmethod
    def run(
        context: Dict[str, Any],
        model: Any,
        callable: Callable,
        prompt: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Any:
        """
        Runs the prompt chain with the given context, model, callable, prompt, and conversation history.

        Args:
            context (Dict[str, Any]): The context dictionary containing variables for the prompt.
            model (Any): The model to use for generating the prompt output.
            callable (Callable): The callable function that takes the model and formatted prompt as arguments.
            prompt (str): The prompt string with optional placeholders for context variables.
            conversation_history (List[Dict[str, str]], optional): The conversation history as a list of dictionaries
                containing 'query' and 'response' keys. Defaults to None.

        Returns:
            Any: The parsed JSON result of the prompt output.

        Raises:
            ValueError: If there is an error decoding the JSON result.
        """

        if conversation_history:
            context['conversation_history'] = "\n".join([
                f"Q: {turn['query']}\nA: {turn['response']}"
                for turn in conversation_history
            ])
        else:
            context['conversation_history'] = ""

        template = Template(prompt)
        formatted_prompt = template.render(context)

        result = callable(model, formatted_prompt)
        return CustomPromptChain._parse_json(result)
    
    @staticmethod
    def _parse_json(result: str) -> Any:
        """
        Parses the result string as JSON.

        Args:
            result (str): The result string to parse.

        Returns:
            Any: The parsed JSON object, or the plain text if JSON parsing fails.
        
        Raises:
            ValueError: If there is an error decoding the JSON result.
        """
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the plain text result
            return result