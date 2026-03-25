import openai

# TODO: use api_key in OpenAI object
client = openai.OpenAI(
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