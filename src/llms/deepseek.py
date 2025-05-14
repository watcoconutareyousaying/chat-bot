import requests
from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

class DeepSeekChat(BaseChatModel):
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.4
    max_tokens: int = 500
    api_base: str = "https://api.deepseek.com/v1"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        openai_messages = []

        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            openai_messages.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return AIMessage(content=result["choices"][0]["message"]["content"])
        except requests.exceptions.RequestException as e:
            raise ValueError(f"DeepSeek API request failed: {str(e)}")

    def _llm_type(self) -> str:
        return "deepseek-chat"
