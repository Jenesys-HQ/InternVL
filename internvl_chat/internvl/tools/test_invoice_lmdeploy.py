import argparse
import json
import time

import fire
import torch
from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

from internvl.tools.data_utils import extract_json_data
from internvl.tools.prompt import prompt


def run_main(ckpt_dir: str, tp: int = 1, cache_max_entry_count: float = .8):
    img_path = "/workspace/test_invoice.jpeg"
    image = load_image(img_path)
    engine_config = TurbomindEngineConfig(
        dtype=torch.bfloat16,
        session_len=45000,
        cache_max_entry_count=cache_max_entry_count,
        tp=tp,
        device_type="cuda"
    )
    pipe = pipeline(ckpt_dir, backend_config=engine_config)
    generation_config = GenerationConfig(
        top_k=50,
        top_p=0.9,
        max_new_tokens=2048,
        random_seed=42
    )

    start = time.perf_counter()
    response = pipe((prompt, image), gen_config=generation_config)
    predicted_data_row = extract_json_data(response.text)
    end = time.perf_counter()

    print(json.dumps(predicted_data_row, indent=4))
    print(f"Time taken for inference: {end - start} seconds")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main()
