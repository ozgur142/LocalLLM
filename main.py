import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set a manual seed for reproducibility
torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize conversation history
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

# Initialize the pipeline for text generation with streaming enabled
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    config={"stream": True}  # Enable streaming
)

# Define generation arguments
generation_args = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "temperature": 0.2,
    "do_sample": True,
}

# Chat loop
print("Start chatting with the model (type 'exit' to stop):")
while True:
    # Get user input
    user_input = input("\nYou: ")

    # Exit the chat loop if the user types 'exit'
    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    # Append user input to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Generate a response from the model
    for output in pipe(messages, **generation_args):
        # Get the model's response and append it to the conversation history
        model_response = output['generated_text'].strip()
        print(f"Model: {model_response}", end='', flush=True)  # Print response as it's generated

    # Append final response to the messages history
    messages.append({"role": "assistant", "content": model_response})
