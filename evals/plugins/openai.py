from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
)
from .base import _ChatCompletionModel, _CompletionModel


class _OpenAICompletionModel(_CompletionModel):
    def run(self, prompt: OpenAICreatePrompt, **kwargs):
        return openai_completion_create_retrying(
            model=self.name, prompt=prompt, **kwargs
        )


class _OpenAIChatCompletionModel(_ChatCompletionModel):
    def run(self, messages: OpenAICreateChatPrompt, **kwargs):
        return openai_chat_completion_create_retrying(
            model=self.name, messages=messages, **kwargs
        )
