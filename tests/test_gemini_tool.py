import asyncio

from api.gemini_client import GeminiClient
from core import config


async def run_tests():
    api_key = config.GEMINI_API_KEY
    url = config.GEMINI_API_LINK
    config_path = config.TOOLS_JSON_PATH

    client = GeminiClient(api_key=api_key, url=url, config_path=config_path)

    try:
        result1 = await client.parse_intent("Hi, nice to meet you, John.")
        assert result1 is not None, "Expected a tool call, but got None"
        assert result1["cmd"] == "REGISTER_FACE", f"Expected REGISTER_FACE, got {result1.get('cmd')}"
        assert result1["args"]["name"] == "John", f"Expected 'John', got {result1['args'].get('name')}"

        result2 = await client.parse_intent("Please remind me he is allergic to peanuts.")
        assert result2 is not None, "Expected a tool call, but got None"
        assert result2["cmd"] == "SAVE_MEMORY", f"Expected SAVE_MEMORY, got {result2.get('cmd')}"
        assert "allergic to peanuts" in result2["args"]["fact"].lower(), "Fact payload did not match expected string"

        result3 = await client.parse_intent("What is the capital of France?")
        assert result3 is None, f"Expected None for general knowledge question, but got {result3}"

        print("All assertions passed successfully!")

    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_tests())
