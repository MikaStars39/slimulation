
def test_prepare_pass_at_k_jsonl():
    from src.data.data import prepare_pass_at_k_jsonl
    # The input configuration string
    config_str = "aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
    output_file = "outputs/debug/prepared_inference_data.jsonl"
    cache_dir = "/mnt/llm-train/users/explore-train/qingyu/.cache"
    
    prepare_pass_at_k_jsonl(
        config_str=config_str, 
        output_file=output_file, 
        cache_dir=cache_dir,
    )
