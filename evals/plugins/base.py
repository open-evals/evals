from abc import ABC, abstractmethod
from typing import ClassVar
import pydantic

from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt


class _ChatCompletionModel(pydantic.BaseModel, ABC):
    name: ClassVar[str]

    @abstractmethod
    def run(self, prompt: OpenAICreatePrompt, **kwargs):
        raise NotImplementedError


class _CompletionModel(pydantic.BaseModel, ABC):
    name: ClassVar[str]

    @abstractmethod
    def run(self, message: OpenAICreateChatPrompt, **kwargs):
        raise NotImplementedError
