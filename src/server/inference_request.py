from openai import OpenAI

client = OpenAI(
    base_url="https://api.z.ai/api/coding/paas/v4",
    api_key="91864271140d47dc9ddc6c2efa69da00.RcwdHFSNnSHP1gVK"
)

class LLMRequest:
    def __init__(self, message, model = "glm-4.5-air"):
        self.message = message
        self.model = model

    def generate(self):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user", "content":self.message}]
        )
        print(response.choices[0].message.content)

request = LLMRequest(message="Hello! Who are you?")
result = request.generate()
print(result)