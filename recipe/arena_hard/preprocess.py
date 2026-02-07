import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Logic --------
def _iter_records_from_line(line: str):
    stripped = line.strip()
    if not stripped or stripped in ("[", "]"):
        return
    if stripped.endswith(","):
        stripped = stripped[:-1]
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    length = len(stripped)
    while idx < length:
        while idx < length and stripped[idx].isspace():
            idx += 1
        if idx >= length:
            break
        if stripped[idx] == ",":
            idx += 1
            continue
        obj, end = decoder.raw_decode(stripped, idx)
        yield obj
        idx = end

def prepare_data(
    input_file: str, 
    output_file: str, 
    tokenizer_name: str,
    system_prompt: str = None,
    thinking: bool = False
):
    """
    Wraps the 'prompt' field and adds explicit formatting instructions.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            for data in _iter_records_from_line(line):
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
                    add_generation_prompt=True,
                    thinking=thinking,
                )

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", required=False)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--thinking", required=False, default=False)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer, args.system_prompt, bool(args.thinking))
    print(f"[Pre-process] Done. Inference file ready: {args.output}")