import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaChatResult:
    content: str
    model: str
    raw: dict


class OllamaChatClient:
    """
    Ollama の OpenAI互換APIへプロンプトを送る簡易クライアント。

    送信先:
      http://localhost:11434/v1/chat/completions
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "tinyswallow-vtuber:latest",
        timeout_sec: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> OllamaChatResult:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_sec,
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama API request failed. "
                f"Is Ollama running at {self.base_url}? "
                f"Original error: {exc}"
            ) from exc

        raw = json.loads(response_body)

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response format: {raw}"
            ) from exc

        return OllamaChatResult(
            content=content,
            model=self.model,
            raw=raw,
        )