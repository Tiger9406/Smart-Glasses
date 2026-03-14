import base64
import json

import aiohttp

from core import config_openai


class OpenAIClient:
    def __init__(self, timeout_seconds=20):
        self.api_key = config_openai.OPENAI_API_KEY
        self.url = config_openai.OPENAI_API_LINK
        self.max_output_tokens = config_openai.MAXOUTPUTTOKENS
        self.temperature = config_openai.TEMPERATURE
        self.model = config_openai.MODEL
        self.vision_model = config_openai.VISION_MODEL

        self.user_name = config_openai.USER_NAME

        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = None
        with open(config_openai.TOOLS_JSON_PATH, "r") as f:
            tools_config = json.load(f)

        self.sys_instruct = (
            tools_config.get("systemInstruction", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if self.sys_instruct:
            self.sys_instruct = self.sys_instruct.format(user_name=self.user_name)

        self.tools = tools_config.get("tools", [])

        with open(config_openai.PROMPT_JSON_PATH, "r") as f:
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
        print("[OPENAI] Session closed")

    async def analyze_video_frames(self, frames: list[bytes], prompt: str) -> str:
        """
        Encode frames & send to OPENAI via http
        return text response or raises exception
        """

        if not self.api_key:
            raise ValueError("API Key not configured")

        content = [{"type": "text", "text": prompt}]

        for jpg_bytes in frames:
            b64_data = base64.b64encode(jpg_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                }
            )

        payload = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        session = await self._get_session()
        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        ) as response:
            return await self._handle_response(response, "VLM")

    async def analyze_memory(
        self, conversation_history: str, known_facts: str = "None"
    ):
        if not self.api_key:
            raise ValueError("API Key not configured")

        prompt_template = self.prompt_config.get("memory_prompt_template", "")
        prompt = prompt_template.format(
            conversation_history=conversation_history, known_facts=known_facts
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a memory analyzer. Output strictly in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        session = await self._get_session()
        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        ) as response:
            return await self._handle_response(response, "MEMORY_JSON")

    async def parse_intent(self, prompt: str) -> str:

        if not self.api_key:
            raise ValueError("API Key not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.sys_instruct},
                {"role": "user", "content": prompt},
            ],
            "tools": self.tools,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        session = await self._get_session()
        async with session.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        ) as response:
            return await self._handle_response(response, "INTENT")

    async def _handle_response(self, response, response_type: str):
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {text}")

        result = await response.json()
        choices = result.get("choices", [])
        if not choices:
            raise ValueError(f"No choices returned. Full response: {result}")

        message = choices[0].get("message", {})

        if response_type == "VLM":
            return message.get("content", "")

        elif response_type == "MEMORY_JSON":
            try:
                return json.loads(message.get("content", "{}"))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON memory: {message.get('content')}")
                return []

        elif response_type == "INTENT":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                func = tool_calls[0].get("function", {})
                return {
                    "cmd": func.get("name", "").upper(),
                    "args": json.loads(func.get("arguments", "{}")),
                }
            elif message.get("content"):
                return {"cmd": "CHAT", "args": {"message": message.get("content")}}
            return None
