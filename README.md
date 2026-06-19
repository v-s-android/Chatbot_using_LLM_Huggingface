Create a Python virtual environment

```
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```

Install libs:
```
pip install transformers==4.41.2 torch==2.2.2 accelerate==0.30.1 numpy==1.26.4
```

For chatbot_llm.py
- You can experiment with different settings to see how the bot’s behavior changes. This is similar to how real AI systems are tuned to feel more friendly, creative, or strict depending on the use case. For example, you can make the bot more friendly by changing the system prompt:
```
messages = [{
    "role": "system",
    "content": "You are a very friendly and cheerful assistant. Always respond in a warm, casual, and encouraging tone."
}]
```
Then try changing generation parameters to see the difference in responses:
```
temperature=0.9
top_p=0.95
```
