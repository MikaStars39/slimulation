import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Logic --------
def prepare_data(
    input_file: str, 
    output_file: str, 
    tokenizer_name: str,
    system_prompt: str = None
):
    """
    Wraps the 'prompt' field and adds explicit formatting instructions.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Formulating the message with the new instructions
            data['prompt'] = [
                {
                    "role": "user", 
                    "content": data["prompt"]
                }
            ]

            if system_prompt is not None:
                data['prompt'].insert(
                    0,
                    {
                        "role": "system", 
                        "content": system_prompt
                    }
                )

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", required=False)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")