import openai
client = openai.OpenAI(
    api_key="sk-4e7h5BLXEa-N5FX430dmHw",
    base_url="https://llm.pubmatic.com"
)
response = client.chat.completions.create(
    model="(paid) gpt-4o-mini", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)
print(response)