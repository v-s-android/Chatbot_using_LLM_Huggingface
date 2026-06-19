from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

# Choose a modern LLM
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading model...")
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map="cpu",
  torch_dtype=torch.float32
)

# Initialize conversation messages
'''
 In modern chat-based LLMs, we use a structured conversation format made of messages.
 Each message has a specific role that tells the model who is speaking and 
 how to behave.
'''
messages = [
  {
      "role": "system",
      "content": "You are a helpful AI assistant. Give short and concise answers in 2-3 lines."
  }
]

# Start chatbot loop
print("Chatbot started. Type 'exit' to quit.\n")
while True:
  user_input = input("> ")

  if user_input.lower() == "exit":
      break
  
  # Update conversation history
  messages.append({"role": "user", "content": user_input})

  # To avoid very long conversations, keep only recent exchanges:
  messages = [messages[0]] + messages[-10:]

  # Apply chat template: Modern Hugging Face chat models use chat templates to format conversations automatically.
  tokenized = tokenizer.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
      max_length=512
  )

  # Generate response from a language model 
  with torch.inference_mode():
      outputs = model.generate(
          tokenized["input_ids"],
          attention_mask=tokenized["attention_mask"],
          max_new_tokens=60,
          temperature=0.5,
          top_p=0.8,
          do_sample=True,
          repetition_penalty=1.3,
          no_repeat_ngram_size=3,
          pad_token_id=tokenizer.pad_token_id
      )

  # Decode and display response
  '''
   After the model generates output, it is still in token form (numbers).
   This step converts it back into readable text and shows it to the user.
  '''  
  response = tokenizer.decode(
      outputs[0][tokenized["input_ids"].shape[-1]:],
      skip_special_tokens=True
  )

  print(f"Bot: {response}\n")
  
  # Save assistant response
  messages.append({"role": "assistant", "content": response})
  