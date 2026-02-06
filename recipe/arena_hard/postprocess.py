"""
Post-processing: Add metadata to model responses for Arena-Hard-Auto evaluation.

This script calculates:
- token_len: Number of tokens in the response
- header_count: Count of markdown headers (h1-h6)
- list_count: Count of ordered/unordered lists
- bold_count: Count of bold text elements

Usage:
    python postprocess.py --input results.jsonl --output final.jsonl \
        --model-name "my-model-v1" --tokenizer /path/to/tokenizer
"""

import json
import argparse
import re
import os
from typing import Dict

import tiktoken
from transformers import AutoTokenizer


def remove_code_blocks(text: str) -> str:
    """
    Remove code blocks (```...```) from text before counting markdown elements.
    This prevents code content from being counted as markdown.
    """
    pattern = re.compile(r"```[^`]*```", re.DOTALL)
    return pattern.sub("", text)


def count_markdown_elements(text: str) -> Dict:
    """
    Count markdown formatting elements in the text.
    
    Args:
        text: Markdown text (with code blocks already removed)
    
    Returns:
        Dictionary with header_count, list_count, bold_count
    """
    counters = {
        "header_count": {
            "h1": len(re.findall(r"^#{1}\s", text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", text, re.MULTILINE)),
        },
        "list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", text, re.MULTILINE)),
        },
        "bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", text)),
            "__": len(re.findall(r"__[^_\n]+__", text)),
        },
    }
    return counters


def calculate_token_length(text: str, tokenizer) -> int:
    """
    Calculate the number of tokens in the text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance (HuggingFace or tiktoken)
    
    Returns:
        Number of tokens
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        # tiktoken tokenizer (for GPT models)
        return len(tokenizer.encode(text, disallowed_special=()))
    else:
        # HuggingFace tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))


def create_metadata(response: str, tokenizer) -> Dict:
    """
    Create complete metadata for a response.
    
    Args:
        response: Model's response text
        tokenizer: Tokenizer instance
    
    Returns:
        Dictionary with all metadata fields
    """
    # Calculate token length
    token_len = calculate_token_length(response, tokenizer)
    
    # Remove code blocks before counting markdown elements
    text_without_code = remove_code_blocks(response)
    
    # Count markdown elements
    markdown_counts = count_markdown_elements(text_without_code)
    
    # Combine all metadata
    metadata = {"token_len": token_len}
    metadata.update(markdown_counts)
    
    return metadata


def add_metadata_to_response(input_data: Dict, model_name: str, tokenizer) -> Dict:
    """
    Add metadata to inference output without changing the original format.
    
    Args:
        input_data: Dict with 'uid', 'prompt', 'response', etc.
        model_name: Name of the model
        tokenizer: Tokenizer instance
    
    Returns:
        Original dict with added 'metadata' and 'model' fields
    """
    response = input_data.get("response", "")
    
    # Keep all original fields and add metadata
    output = input_data.copy()
    output["model"] = model_name
    output["metadata"] = create_metadata(response, tokenizer)
    
    return output


def postprocess(input_file: str, output_file: str, model_name: str, 
                tokenizer_name: str, use_tiktoken: bool = False):
    """
    Main post-processing pipeline.
    
    Args:
        input_file: Input JSONL file with inference results
        output_file: Output JSONL file with added metadata
        model_name: Name of the model for identification
        tokenizer_name: Path to tokenizer or tiktoken model name
        use_tiktoken: Whether to use tiktoken (for GPT models)
    """
    # Load tokenizer
    if use_tiktoken:
        print(f"Loading tiktoken encoder: {tokenizer_name}")
        tokenizer = tiktoken.encoding_for_model(tokenizer_name)
    else:
        print(f"Loading HuggingFace tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process inference results
    print(f"Processing inference results: {input_file}")
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Add metadata to existing format
            output_data = add_metadata_to_response(data, model_name, tokenizer)
            
            # Write to output file
            f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"✓ Processed {processed_count} responses")
    print(f"✓ Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process inference results with metadata for Arena-Hard-Auto"
    )
    
    # I/O arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with inference results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file in Arena-Hard-Auto format")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model (e.g., 'my-model-v1')")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer or tiktoken model name")
    parser.add_argument("--use-tiktoken", action="store_true",
                        help="Use tiktoken instead of HuggingFace tokenizer")
    
    args = parser.parse_args()
    
    postprocess(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer,
        use_tiktoken=args.use_tiktoken
    )


if __name__ == "__main__":
    main()
