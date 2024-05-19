import json
from trimit.utils.cache import CacheMixin
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os


class GPTMixin(CacheMixin):
    def __init__(self, cache_prefix="gpt/", cache=None):
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        super().__init__(cache=cache, cache_prefix=cache_prefix)

    @retry(
        stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=60)
    )
    def call_gpt(
        self,
        prompt,
        require_json=False,
        use_existing_output=True,
        prior_messages: list | None = None,
        model: str = "gpt-4o",
        base64_images: list | None = None,
    ):
        if prompt in self.cache and use_existing_output:
            print("USING CACHE")
            res = self.cache[prompt]
            self.cache.close()
            return res
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent subcomponent of a larger webapp designed to create professional, high-quality, narrative videos from text prompts.",
            }
        ]
        if prior_messages:
            messages.extend(prior_messages)
        prompt_content = prompt
        if base64_images:
            prompt_content = [{"type": "text", "text": prompt}]
            for base64_image in base64_images:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    }
                )
        messages.append({"role": "user", "content": prompt_content})

        completion_kwargs = dict(model=model, messages=messages)

        if require_json:
            completion_kwargs["response_format"] = {"type": "json_object"}

        print("Calling GPT")
        completion = self.openai_client.chat.completions.create(**completion_kwargs)
        # Access the content attribute
        content = completion.choices[0].message.content
        self.cache.set(prompt, content)
        self.cache.close()
        finish_reason = completion.choices[0].finish_reason
        if "finish_reason" == "length":
            print("GPT ran out of tokens, finish_reason=='length'")
            return ""
        if content is None:
            print("GPT returned None. Finish reason: ", finish_reason)
            return ""
        return content

    async def get_json_from_gpt_no_history(
        self, prompt, retries_till_parseable=3, use_existing_output=True
    ):
        prompt_addition_template = "Your previous {} responses were not JSON-parseable. Please try again with the same prompt as follows: \n"
        prompt_addition = ""
        for i in range(retries_till_parseable):
            response = self.call_gpt(
                prompt_addition + prompt,
                require_json=True,
                use_existing_output=use_existing_output,
            )
            if not response:
                return {}
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                print(f"Could not get JSON from GPT after {i + 1} tries")
                prompt_addition = prompt_addition_template.format(i + 1)
                continue
            else:
                return parsed
        raise ValueError(
            f"Could not get JSON from GPT after {retries_till_parseable} tries"
        )
