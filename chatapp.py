import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Latest Mistral 7B Instruct model
MAX_NEW_TOKENS = 512  # Maximum number of tokens to generate
MAX_HISTORY_LENGTH = 5  # Maximum number of conversation turns to keep

# Set up quantization configuration for 4-bit loading (minimizes VRAM usage)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Further reduces memory footprint
)

# Load model and tokenizer
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config,
)
print("Model loaded successfully!")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available. Using CPU.")

def format_prompt(message, history=None):
    """Format prompt according to Mistral instruction format."""
    prompt = ""
    
    # Include condensed history if provided
    if history:
        for user_msg, bot_msg in history[-MAX_HISTORY_LENGTH:]:
            prompt += f"<s>[INST] {user_msg} [/INST] {bot_msg}</s>\n"
    
    # Add the current message
    prompt += f"<s>[INST] {message} [/INST]"
    
    return prompt

def generate_response(prompt):
    """Generate a response from the model given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated response (after the prompt)
        response = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        # Clean up any remaining instruction tokens that might be in the response
        response = response.replace("[/INST]", "").strip()
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def respond(message, history):
    """Process the user message and update the conversation history."""
    prompt = format_prompt(message, history)
    response = generate_response(prompt)
    
    # Memory management: explicitly clear CUDA cache every few interactions
    if len(history) % 3 == 0:
        torch.cuda.empty_cache()
    
    return response

# Create Gradio interface with chat history
with gr.Blocks(title="AI Personal Tutor") as demo:
    gr.Markdown("# 🤖 AI Personal Tutor using Mistral 7B")
    gr.Markdown("Ask any question and get personalized tutoring!")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="Ask me anything about math, science, history, or any subject...",
        container=False,
        scale=7
    )
    submit_btn = gr.Button("Send", variant="primary", scale=1)
    
    clear_btn = gr.Button("Clear Chat", scale=1)
    
    def user_input(user_message, history):
        if not user_message:
            return gr.update(value=""), history
        
        bot_response = respond(user_message, history)
        history = history + [[user_message, bot_response]]
        return "", history
    
    msg.submit(user_input, [msg, chatbot], [msg, chatbot])
    submit_btn.click(user_input, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown("""
    ## About this Tutor
    
    This AI tutor is powered by the Mistral 7B Instruct model. It can:
    - Explain complex concepts in simple terms
    - Answer questions across various subjects
    - Provide step-by-step solutions to problems
    
    Note: While the AI tries to provide accurate information, always verify important facts.
    """)

if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True to create a public link