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
    @lru_cache(maxsize=128)
    def _parse_json(json_string: str) -> Union[Dict, List, str]:
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_string)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return json_string

    @staticmethod
    def run(
        context: Dict[str, Any],
        model: Any,
        callable: Callable,
        prompt: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Any:
        """
        Run a single prompt with context and conversation history.
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



