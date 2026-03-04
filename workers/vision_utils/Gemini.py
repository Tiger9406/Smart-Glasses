import base64

import aiohttp

from core.config import MAXOUTPUTTOKENS, TEMPERATURE


class GeminiClient:
    def __init__(self, api_key: str, url: str, timeout_seconds=20):
        self.api_key = api_key
        self.url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = None

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
            return await self._handle_response(response)

    async def _handle_response(self, response):
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

        try:  # get text
            return candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Fallback if structure is valid but content is unexpectedly missing
            # doesnt need to be raised value error cus if its a empty string then there is no text so its fine
            return ""
            raise ValueError(
                f"Valid finishReason but missing text content. Response: {result}"
            )

    async def analyze_memory(self, command):
        if not self.api_key:
            raise ValueError("API Key not configured")

        name = command.get("name", "Unknown")
        facts_string = command.get("facts_string", "")
        speech_text = command.get("speech_text", "")

        prompt = f"""You are a memory extraction module. 
        Current known facts about {name}:
        {facts_string if facts_string else "None."}

        Latest speech from {name}: "{speech_text}"

        Extract any NEW, concrete information worth remembering (preferences, names, intentions). 
        Make sure the information is useful.
        If it contains information about someone else only store it if you have their name.
        Be short and concise. 
        
        IF NO NEW INFORMATION RETURN AN EMPTY STRING."""

        parts = [{"text": prompt}]

        session = await self._get_session()

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAXOUTPUTTOKENS,
            },
        }

        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        ) as response:
            new_facts = await self._handle_response(response)

        return name, new_facts
