import json
import asyncio
import sglang as sgl
from tqdm.asyncio import tqdm
# import aiofiles

async def run_offline_async_inference(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    dp_size: int,
    tp_size: int,
    max_concurrency: int, 
    mem_fraction_static: float,
    sampling_params: dict,
    batch_size: int,
):
    # 1. Initialize Engine
    llm = sgl.Engine(
        model_path=model_path,
        dp_size=dp_size, 
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        log_level="error"
    )

    # 2. Worker Logic (Consumer)
    # 使用 Queue 解耦，限制并发数，防止内存爆炸
    queue = asyncio.Queue(maxsize=max_concurrency * 2)
    
    # 计算总行数用于进度条（同步读取即可，速度很快）
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8') if _.strip())
    pbar = tqdm(total=total_lines, desc=f"Inference (DP={dp_size})")

    async def worker(f_out):
        while True:
            # 从队列获取数据
            item = await queue.get()
            if item is None: 
                # 结束信号
                queue.task_done()
                break
            
            try:
                # 异步推理
                output = await llm.async_generate(item["prompt"], sampling_params)
                item["response"] = output["text"]
                
                # 同步写入 (虽然是同步，但去掉flush后通常非常快，不会成为瓶颈)
                line = json.dumps(item, ensure_ascii=False) + "\n"
                f_out.write(line)
                
            except Exception as e:
                print(f"Error processing item: {e}")
            finally:
                pbar.update(1)
                queue.task_done()

    # 3. Producer Logic (File Reader)
    async def producer():
        # 同步读取文件，但逐行放入异步队列
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    await queue.put(json.loads(line))
        
        # 发送结束信号给所有 worker
        for _ in range(max_concurrency):
            await queue.put(None)

    # 4. Run Pipeline
    # 使用标准的 open
    with open(output_file, "w", encoding="utf-8") as f_out:
        # 启动消费者 (Workers)
        workers = [asyncio.create_task(worker(f_out)) for _ in range(max_concurrency)]
        
        # 启动生产者
        await producer()
        
        # 等待所有任务完成
        await queue.join()
        await asyncio.gather(*workers)

    pbar.close()
    llm.shutdown()
    print(f"Done! Results saved to {output_file}")