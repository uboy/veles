"""LLM-провайдер для синтеза ответов из evidence-pack.

OpenAI-совместимый эндпоинт = любой провайдер (облако, Ollama, llama.cpp, vLLM) -
конфигурация, не код (стандарт v3.0 §6). Конфиг из окружения:
  VELES_LLM_BASE_URL (напр. https://api.deepseek.com/v1 или http://localhost:11434/v1)
  VELES_LLM_MODEL    (напр. hermes-4-8b / deepseek-chat / gemma4-hermes-tools:Q4_K_M)
  VELES_LLM_API_KEY
Если base_url/model не заданы - провайдер выключен, фасад отвечает извлечённым
фрагментом (graceful degradation, АПИ не падает).

Локальный профиль по KB-исследованию Дениса (gemma4-hermes-tools): Hermes tool-calling
модели через Ollama/llama.cpp - тот же OpenAI-совместимый контракт.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import httpx
from pydantic import BaseModel

SYNTHESIS_SYSTEM = (
    "Ты - ассистент базы знаний. Отвечай ТОЛЬКО на основе предоставленных фрагментов. "
    "Каждое фактическое утверждение сопровождай ссылкой на номер фрагмента в квадратных "
    "скобках, например [1]. Если фрагменты не содержат ответа - честно скажи, что "
    "ответа в базе нет, и не придумывай."
)


class LLMConfig(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 60.0


def llm_config_from_env() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ.get("VELES_LLM_BASE_URL", ""),
        model=os.environ.get("VELES_LLM_MODEL", ""),
        api_key=os.environ.get("VELES_LLM_API_KEY", ""),
    )


class OpenAICompatibleProvider:
    """Минимальный chat-completions клиент; streaming не нужен для синтеза ответа."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.base_url and self.config.model)

    def synthesize(self, question: str, evidence_texts: list[str]) -> str:
        if not self.enabled:
            raise RuntimeError("LLM provider не настроен")
        numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts))
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nФрагменты базы знаний:\n{numbered}",
                },
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                r = httpx.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_s,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - деградация вместо падения фасада
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"LLM недоступен: {type(last_err).__name__}: {last_err}")


def request_id() -> str:
    return uuid.uuid4().hex[:16]
