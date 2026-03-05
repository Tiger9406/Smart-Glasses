import base64
import json

import aiohttp

from core import config


class GeminiClient:
    def __init__(self, timeout_seconds=20):
        self.api_key = config.GEMINI_API_KEY
        self.url = config.GEMINI_API_LINK
        self.max_output_tokens = config.MAXOUTPUTTOKENS
        self.temperature = config.TEMPERATURE

        self.user_name = config.USER_NAME

        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = None
        with open(config.TOOLS_JSON_PATH, "r") as f:
            self.tools_config = json.load(f)
        with open(config.PROMPT_JSON_PATH, "r") as f:
            self.prompt_config = json.load(f)

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
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
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

    async def analyze_memory(self, conversation_history: str):
        if not self.api_key:
            raise ValueError("API Key not configured")

        prompt_template = self.prompt_config.get("memory_prompt_template", "")
        prompt = prompt_template.format(conversation_history=conversation_history)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
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
            extracted_data = await self._handle_response(response, "MEMORY_JSON")

        return extracted_data

    async def parse_intent(self, prompt: str) -> str:

        if not self.api_key:
            raise ValueError("API Key not configured")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": self.tools_config.get("systemInstruction"),
            "tools": self.tools_config.get("tools"),
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "temperature": self.temperature,
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
                if finish_reason == "STOP":
                    return ""
                raise ValueError(
                    f"Valid finishReason but missing text content. Response: {result}"
                )
        elif response_type == "MEMORY_JSON":
            try:
                text = candidate["content"]["parts"][0]["text"]
                return json.loads(text.strip())
            except (KeyError, IndexError):
                if finish_reason == "STOP":
                    return []  # empty list if it stopped normally but found nothing
                raise ValueError(
                    f"Valid finishReason but missing text content. Response: {result}"
                )
            except json.JSONDecodeError:
                print(f"Failed to parse JSON memory: {text}")
                return []
        elif response_type == "INTENT":
            try:
                part = candidate.get("content", {}).get("parts", [])[0]
                if "functionCall" in part:
                    func_call = part["functionCall"]
                    return {
                        "cmd": func_call["name"].upper(),
                        "args": func_call.get("args", {}),
                    }
                elif "text" in part:
                    return {"cmd": "CHAT", "args": {"message": part["text"]}}
                return None
            except (KeyError, IndexError):
                # error here; if no output sohuld be none
                raise ValueError(
                    f"Valid finishReason but missing text content. Response: {result}"
                )
