import openai
import httpx
import pandas as pd
from tqdm import tqdm
import argparse
import concurrent.futures
import logging
import time
import subprocess
import signal
import os
import random
import numpy as np
import torch
from typing import List, Dict, Any

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_available_gpus(count: int = None) -> List[int]:
    """
    Get list of available GPU IDs.
    Same as before: checks system for NVIDIA GPUs.
    """
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        if count is None or count > total_gpus:
            count = total_gpus
        return list(range(count))
    return []

class SGLangServerManager:
    """
    Manages multiple SGLang server instances.
    Replaces the previous VLLMServerManager.
    """
    def __init__(self, model_name: str, start_port: int, gpu_ids: List[int]):
        self.model_name = model_name
        self.start_port = start_port
        self.gpu_ids = gpu_ids
        self.processes: Dict[int, subprocess.Popen] = {} # port -> process object
        self.port_mapping: Dict[int, int] = {} # gpu_id -> port

    def start_all(self):
        """Start SGLang servers on all assigned GPUs."""
        logging.info(f"Starting {len(self.gpu_ids)} SGLang servers...")
        for i, gpu_id in enumerate(self.gpu_ids):
            port = self.start_port + i
            self.port_mapping[gpu_id] = port
            self._start_server(gpu_id, port)
        
        # Wait until all servers pass the health check
        if not self._wait_for_all_servers():
            raise RuntimeError("Some SGLang servers failed to start.")

    def _start_server(self, gpu_id: int, port: int):
        # Set environment variable to isolate the GPU for this process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Construct the SGLang launch command based on documentation
        # Note: SGLang uses --model-path instead of --model in some versions, 
        # but often both work. We follow the provided doc: python3 -m sglang.launch_server
        command = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_name,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--tp-size", "1", # Tensor Parallel size = 1 (since we do Data Parallel manually)
            "--log-level", "info" # Use 'warning' to reduce noise if needed
        ]
        
        # Add optional memory argument if needed (SGLang uses mem-fraction-static)
        # command.extend(["--mem-fraction-static", "0.90"])

        logging.info(f"Launching SGLang on GPU {gpu_id} (Port {port})...")
        
        # Run the command in background
        self.processes[port] = subprocess.Popen(
            command, 
            env=env,
            stdout=subprocess.DEVNULL, # Hide standard output to keep console clean
            stderr=subprocess.PIPE     # Capture errors
        )

    def _wait_for_all_servers(self, timeout: int = 600) -> bool:
        """Wait for /health endpoint to return 200 OK using concurrent checks."""
        start_time = time.time()
        ready_ports = set()
        
        def check_port(port: int) -> bool:
            try:
                with httpx.Client(timeout=1) as client:
                    return client.get(f"http://127.0.0.1:{port}/health").status_code == 200
            except httpx.RequestError:
                return False
        
        while time.time() - start_time < timeout:
            if len(ready_ports) == len(self.processes):
                logging.info("All SGLang servers are ready!")
                return True

            # Check for dead processes first
            for port, proc in self.processes.items():
                if port not in ready_ports and proc.poll() is not None:
                    _, stderr = proc.communicate()
                    logging.error(f"Server on port {port} died. Stderr: {stderr.decode()}")
                    return False

            # Concurrent health checks for pending ports
            pending = [p for p in self.processes if p not in ready_ports]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {executor.submit(check_port, p): p for p in pending}
                for future in concurrent.futures.as_completed(futures):
                    port = futures[future]
                    if future.result():
                        ready_ports.add(port)
                        logging.info(f"Server localhost:{port} is ready ({len(ready_ports)}/{len(self.processes)})")
            
            time.sleep(1)
        
        logging.error("Timeout waiting for SGLang servers.")
        return False

    def stop_all(self):
        """Kill all background server processes."""
        logging.info("Stopping all servers...")
        for port, proc in self.processes.items():
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except Exception:
                pass
        for proc in self.processes.values():
            proc.wait()

class ShardProcessor:
    """
    Worker logic: Sends requests to the SGLang server.
    SGLang is OpenAI-Compatible, so we can keep using openai python client.
    """
    def __init__(
        self, 
        port: int,
        model_name: str, 
        max_tokens: int = 1024, 
        temperature: float = 0.6, 
        top_p: float = 1.0, 
        n_samples: int = 1, 
        api_key: str = "EMPTY"
    ):
        # SGLang's OpenAI-compatible endpoint is usually at /v1
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.api_key = api_key or "EMPTY" # SGLang usually doesn't need a real key locally
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples
        
        # Initialize OpenAI Client with larger connection pool for high concurrency
        self.client = openai.OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=None, 
            max_retries=3,
            http_client=httpx.Client(limits=httpx.Limits(max_connections=500, max_keepalive_connections=100))
        )

    def _request_api(self, prompt: str) -> List[str]:
        try:
            # Using standard ChatCompletion API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role":"system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n_samples,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logging.warning(f"API Error on prompt '{prompt[:20]}...': {e}")
            return []

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item (row) using 'prompt' field."""
        prompt = item.get('prompt', '')
        
        if prompt:
            item['responses'] = self._request_api(prompt)
            
        return item

def process_shard(
    data_shard: List[Dict[str, Any]], 
    port: int, 
    model_name: str,
    gpu_id: int,
    workers_per_gpu: int = 64,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 1.0,
    n_samples: int = 1,
    api_key: str = "EMPTY"
) -> List[Dict[str, Any]]:
    """
    Run processing for a specific GPU shard.
    """
    processor = ShardProcessor(
        port=port, 
        model_name=model_name, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        n_samples=n_samples, 
        api_key=api_key
    )
    results = []
    
    # Thread pool for concurrent requests to the same server
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_per_gpu) as executor:
        futures = {executor.submit(processor.process_item, item): item for item in data_shard}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(data_shard), 
                           desc=f"GPU {gpu_id} (Port {port})", 
                           position=gpu_id, 
                           leave=False):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Item processing failed: {e}")
                
    return results

def run_online_inference(
    input_file: str,
    output_file: str,
    model_name: str,
    num_gpus: int = None,
    start_port: int = 30000,
    workers_per_gpu: int = 64,
    n_samples: int = 1,
    temperature: float = 0.6,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    api_key: str = "EMPTY"
):
    # --------------------------- 1. Load Data --------------------------- 

    logging.info(f"Loading data from: {input_file}")
    df = pd.read_json(input_file, lines=True)
    all_data = df.to_dict('records')
    random.shuffle(all_data)  # Shuffle to balance load roughly
     
    # ---------------------------  2. Detect GPUs and Split Data --------------------------- 
    gpu_ids = get_available_gpus(num_gpus)
    if not gpu_ids:
        raise RuntimeError("No GPUs found!")
    
    num_shards = len(gpu_ids)
    logging.info(f"Detected {num_shards} GPUs. Starting {num_shards} SGLang instances.")
    
    data_shards = np.array_split(all_data, num_shards)
    data_shards = [shard.tolist() for shard in data_shards]

    # ---------------------------  3. Start SGLang Servers --------------------------- 
    server_manager = SGLangServerManager(model_name=model_name, start_port=start_port, gpu_ids=gpu_ids)
    server_manager.start_all()
    
    try:
        # ---------------------------  4. Run Inference in Parallel --------------------------- 
        final_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
            future_to_shard = {
                executor.submit(
                    process_shard, 
                    data_shard=data_shards[i], 
                    port=server_manager.port_mapping[gpu_id], 
                    model_name=model_name, 
                    gpu_id=gpu_id,
                    workers_per_gpu=workers_per_gpu,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n_samples=n_samples,
                    api_key=api_key
                ): gpu_id
                for i, gpu_id in enumerate(gpu_ids) if len(data_shards[i]) > 0
            }
            
            logging.info("All tasks distributed. Inferencing...")
            
            for future in concurrent.futures.as_completed(future_to_shard):
                gpu = future_to_shard[future]
                try:
                    final_results.extend(future.result())
                    logging.info(f"GPU {gpu} finished.")
                except Exception as e:
                    logging.error(f"GPU {gpu} failed: {e}")

        total_time = time.time() - start_time
        logging.info(f"Finished. Total time: {total_time:.2f}s, Speed: {len(all_data)/total_time:.2f} items/s")

        # ---------------------------  5. Save Results --------------------------- 
        if final_results:
            out_df = pd.DataFrame(final_results)
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            out_df.to_json(output_file, orient='records', lines=True)
            logging.info(f"Saved to {output_file}")

    finally:
        server_manager.stop_all()

def main():
    parser = argparse.ArgumentParser(description="Data Parallel SGLang Inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input json/jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_name", type=str, required=True, help="Model path or HuggingFace repo ID")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--start_port", type=int, default=30000, help="Starting port number")
    parser.add_argument("--workers_per_gpu", type=int, default=64, help="Concurrency per GPU")
    
    # Sampling parameters
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--api_key", type=str, default="EMPTY")

    args = parser.parse_args()
    
    run_online_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        num_gpus=args.num_gpus,
        start_port=args.start_port,
        workers_per_gpu=args.workers_per_gpu,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main()