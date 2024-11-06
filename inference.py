import torch
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json
import argparse
from utils import get_bnb_config, get_prompt
import bitsandbytes as bnb


def main():
    # Argument parser to take parameters from run.sh script
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint folder.",
    )
    parser.add_argument(
        "--peft_path", type=str, required=True, help="Path to the adapter checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the input file (.json).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file (.json).",
    )

    args = parser.parse_args()

    # Load tokenizer and model from the saved checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path
    )

    bnb_config = get_bnb_config()
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, quantization_config=bnb_config, device_map="auto"
    )

    # Load Peft model
    model = PeftModel.from_pretrained(base_model, args.peft_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    with open(args.test_data_path, "r") as f:
        private_test_data = json.load(f)

    # Extract instructions from the dataset for inference
    instructions = [get_prompt(item["instruction"]) for item in private_test_data]
    
    predictions = []

    for i, instruction in enumerate(tqdm(instructions, desc="Processing Instructions")):
        # Prepare input tensor for model
        inputs = tokenizer(instruction, return_tensors="pt").to(device)

        # Generate output using the model
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=512,  # Adjust max_length if needed
                num_beams=15,  # Beam search can improve results
                early_stopping=True,
            )

        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract only the part after "ASSISTANT:" in the generated output
        assistant_text = generated_text.split("ASSISTANT:")[-1].strip()

        # Append the result to predictions list
        predictions.append({"id": private_test_data[i]["id"], "output": assistant_text})

    # Save predictions to a JSON file
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print(f"Inference completed and predictions saved to {args.output_file}.")


if __name__ == "__main__":
    main()
