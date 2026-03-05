import base64
import json

import aiohttp

from core.config import MAXOUTPUTTOKENS, TEMPERATURE, TOOLS_JSON_PATH


class GeminiClient:
    def __init__(
        self, api_key: str, url: str, config_path=TOOLS_JSON_PATH, timeout_seconds=20
    ):
        self.api_key = api_key
        self.url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = None
        with open(config_path, "r") as f:
            self.tools_config = json.load(f)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=False)
            self._session = aiohttp.ClientSession(
                timeout=self.timeout, connector=connector
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        print("[VLM] Session closed")

    async def analyze_video_frames(self, frames: list[bytes], prompt: str) -> str:
        """
        Encode frames & send to gemini via http
        return text response or raises exception
        """

        if not self.api_key:
            raise ValueError("API Key not configured")

        parts = [{"text": prompt}]

        for jpg_bytes in frames:
            b64_data = base64.b64encode(jpg_bytes).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_data}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAXOUTPUTTOKENS,
            },
        }

        session = await self._get_session()

        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        ) as response:
            return await self._handle_response(response, "VLM")

    async def parse_intent(self, prompt: str) -> str:

        if not self.api_key:
            raise ValueError("API Key not configured")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": self.tools_config.get("systemInstruction"),
            "tools": self.tools_config.get("tools"),
            "generationConfig": {
                "temperature": 0.1,
            },
        }

        session = await self._get_session()

        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        ) as response:
            return await self._handle_response(response, "INTENT")

    async def _handle_response(self, response, response_type: str):
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {text}")

        result = await response.json()

        prompt_feedback = result.get("promptFeedback", {})
        if prompt_feedback.get("blockReason"):  # blocked
            raise ValueError(f"Prompt blocked: {prompt_feedback['blockReason']}")

        candidates = result.get("candidates", [])
        if not candidates:  # complete blcok or filter
            raise ValueError(f"No candidates returned. Full response: {result}")

        candidate = candidates[0]

        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason != "STOP":  # may have stopped bc safety
            safety_ratings = candidate.get("safetyRatings", [])
            raise ValueError(
                f"Generation stopped due to: {finish_reason}. Safety Ratings: {safety_ratings}"
            )

        if response_type == "VLM":
            try:  # get text
                text = candidate["content"]["parts"][0]["text"]
                return text
            except (KeyError, IndexError):
                # Fallback if structure is valid but content is unexpectedly missing
                raise ValueError(
                    f"Valid finishReason but missing text content. Response: {result}"
                )
        elif response_type == "INTENT":
            try:
                part = candidate.get("content", {}).get("parts", [])[0]
                if "functionCall" in part:
                    func_call = part["functionCall"]
                    return {
                        "cmd": func_call["name"].upper(),
                        "args": func_call.get("args", {}),
                    }
                return None
            except (KeyError, IndexError):
                # Fallback if structure is valid but content is unexpectedly missing
                raise ValueError(
                    f"Valid finishReason but missing text content. Response: {result}"
                )
