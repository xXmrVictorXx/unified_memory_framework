import base64
from openai import OpenAI

img_path = "/home/eg4/lsm_test/cat.png"
model_name = "Qwen2.5-VL-72B-Instruct"

client = OpenAI(
    base_url="http://127.0.0.1:18002/v1",
    api_key="no-api-key-needed"  # vLLM 不需要 API 密钥
)

message = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请描述这张图片的内容。"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')}"
                }
            }
        ]
    }
]

response = client.chat.completions.create(
    model=model_name,
    messages=message,
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)