import gradio as gr
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
API_URL = "<your_api_url_here>/generate" # We hosted the model in the personal server
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95

def create_payload(message, max_tokens, temperature, top_p):
    """Create the payload for the API request."""
    return {
        "prompt": message,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

def call_api(payload):
    """Make the API call and return the response."""
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.RequestException as e:
        logging.error(f"API call failed: {e}")
        return None

def extract_generated_text(api_response):
    """Extract the generated text from the API response."""
    if api_response and 'generated_text' in api_response:
        return api_response['generated_text']
    logging.warning("Generated text not found in API response")
    return None

def respond(message, history, max_tokens, temperature, top_p):
    """Generate a response using the API."""
    payload = create_payload(message, max_tokens, temperature, top_p)
    logging.info(f"Sending payload: {payload}")

    api_response = call_api(payload)
    if api_response:
        generated_text = extract_generated_text(api_response)
        if generated_text:
            yield generated_text
        else:
            yield "Error: Unable to extract generated text from API response."
    else:
        yield "Error: Unable to get response from API."

def create_chat_interface():
    """Create and configure the Gradio chat interface."""
    return gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Slider(minimum=50, maximum=2048, value=DEFAULT_MAX_TOKENS, step=1, label="Max new tokens"),
            gr.Slider(minimum=0.1, maximum=4.0, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature"),
            gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P, step=0.05, label="Top-p (nucleus sampling)"),
        ],
    )

def main():
    demo = create_chat_interface()
    demo.launch()

if __name__ == "__main__":
    main()