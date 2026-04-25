import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen 0.5B Model... this might take a minute on boot.")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype="auto",
    device_map="auto"
)

def generate_reply(user_message: str) -> str:
    """
    Takes a user's message, formats it for the chat model, and returns the generated string.
    """
    messages = [
        {"role": "system", "content": "You are CommunityBot, the AccessAbility Peer Network's Discord companion.  We made you!"},
        {"role": "user", "content": user_message}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,       # Limit the length of the reply
        temperature=0.7,          # Creativity (lower = more focused, higher = more random)
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()

print("Model loaded successfully!")