import base64

import aiohttp

from core.config import TEMPERATURE, MAXOUTPUTTOKENS


class VLMClient:
    def __init__(self, api_key: str, url: str):
        self.api_key = api_key
        self.url = url

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
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAXOUTPUTTOKENS},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
            ) as response:
                return await self._handle_response(response)

    async def _handle_response(self, response):
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {text}")

        result = await response.json()

        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected JSON structure: {result}")
