import asyncio

from api.gemini_client import GeminiClient


async def run_memory_tests():
    client = GeminiClient()

    try:
        print("--- Running Memory Analysis Tests ---")

        # --- Test Case 1: Standard Extraction ---
        print("\nTesting Case 1: Extracting New Facts...")
        convo_1 = (
            "[Alice] I just got back from my trip to Japan, it was amazing.\n"
            "[Bob] Oh nice! I am incredibly allergic to sushi, so I've never gone."
        )
        known_facts_1 = "None"

        result1 = await client.analyze_memory(convo_1, known_facts=known_facts_1)

        assert isinstance(result1, list), f"Expected a list, got {type(result1)}"
        subjects_1 = [item.get("subject") for item in result1]
        assert "Alice" in subjects_1, "Failed to extract facts for Alice"
        assert "Bob" in subjects_1, "Failed to extract facts for Bob"
        print(f"Case 1 Passed! Extracted:\n{result1}")

        # --- Test Case 2: Deduplication (Ignoring Known Facts) ---
        print("\nTesting Case 2: Deduplication with Known Facts...")
        convo_2 = (
            "[Alice] Like I said earlier, my trip to Japan was great.\n"
            "[Bob] Yeah, I remember. Also, I forgot to mention, I just adopted a dog named Buster."
        )
        # We format the known facts exactly as you suggested
        known_facts_2 = "[Alice] Took a trip to Japan."

        result2 = await client.analyze_memory(convo_2, known_facts=known_facts_2)

        assert isinstance(result2, list), f"Expected a list, got {type(result2)}"

        # Check that the model extracted Bob's new dog
        facts_text = " ".join([item.get("fact", "").lower() for item in result2])
        assert "dog" in facts_text or "buster" in facts_text, (
            "Failed to extract Bob's new dog."
        )

        # Verify the model IGNORED Alice's trip to Japan because it was in known_facts
        japan_mentioned = any(
            "japan" in item.get("fact", "").lower()
            for item in result2
            if item.get("subject") == "Alice"
        )
        assert not japan_mentioned, (
            "Model failed to deduplicate! It extracted the Japan fact again."
        )

        print(
            f"Case 2 Passed! Successfully ignored known facts and extracted:\n{result2}"
        )

        print("\nAll memory analysis assertions passed successfully!")

    except AssertionError as e:
        print(f"\nAssertion Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_memory_tests())
