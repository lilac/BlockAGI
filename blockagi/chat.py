"""ChatModel via DuckDuckGo's chat API."""
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import BaseMessage
from duckduckgo_search import DDGS
from pydantic import Field

class DDGChatModel(SimpleChatModel):
    """ChatModel via DuckDuckGo's chat API."""

    model_name: str = Field(default="gpt-3.5", alias="model")
    """The model to use: "gpt-3.5", "claude-3-haiku". Defaults to "gpt-3.5"."""

    @property
    def _llm_type(self) -> str:
        return "duck-duck-go-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # todo: support multiple messages
        session = DDGS()
        prompt = "\n".join(map(lambda m: m.content, messages))
        response = session.chat(prompt, self.model_name)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
