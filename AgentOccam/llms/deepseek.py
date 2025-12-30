from openai import OpenAI
import time
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def call_deepseek(prompt, model_id="deepseek-v3.2", system_prompt=DEFAULT_SYSTEM_PROMPT):
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("DeepSeek request failed.")
        try:
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY
            )

            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.95,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                extra_body={"reasoning": {"enabled": True}}
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1


def arrange_message_for_deepseek(item_list):
    prompt = "".join([item[1] for item in item_list if item[0] == "text"])
    return prompt


def call_deepseek_with_messages(messages, model_id="deepseek-v3.2", system_prompt=DEFAULT_SYSTEM_PROMPT):
    return call_deepseek(prompt=messages, model_id=model_id, system_prompt=system_prompt)
