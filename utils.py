from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    """Format the instruction as a prompt for LLM."""
    return f"請你扮演一個人工智慧國文助理，幫助用戶在白話文和文言文之間轉換，USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    """Get the BitsAndBytesConfig."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config
