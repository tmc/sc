
def format_prompt(messages):
    """
    Formats a list of messages into a chat prompt.
    Adapts to Qwen/Llama formats if needed, or uses a simple custom format.
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant:\n{content}\n"
    prompt += "Assistant:\n"
    return prompt
