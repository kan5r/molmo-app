import argparse
import gradio as gr
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from utils import resize_image, get_coords, show_points, plot_image
from PIL import Image



def load_transformers_model():
    try:
        processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
        return processor, model, "transformers"
    except Exception as e:
        print(f"Error during transformers model loading: {e}")
        return None, None, None


def load_vllm_model():
    try:
        processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
        model = LLM(
            model="allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            dtype="bfloat16"
        )
        return processor, model, "vllm"
    except Exception as e:
        print(f"Error during vLLM model loading: {e}")
        return None, None, None


def generate_transformers(processor, model, image, text, temperature=0.0, max_new_tokens=1024):
    if image is None:
        return "Please upload an image."

    try:
        inputs = processor.process(
            images=[image],
            text=text
        )
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):  # Use autocast for efficiency
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens, 
                    stop_strings="<|endoftext|>", 
                    do_sample=True if temperature > 0.0 else False,
                    temperature=temperature if temperature > 0.0 else None
                ),
                tokenizer=processor.tokenizer
            )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    except Exception as e:
        return f"Error during generation: {e}"


def generate_vllm(processor, model, image, text, temperature=0.0, max_new_tokens=1024):
    if image is None:
        return "Please upload an image."
    
    try:
        inputs_dict = {"prompt": text, "multi_modal_data": {"image": image}}
        output = model.generate(
            [inputs_dict],
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=["<|endoftext|>"]
            )
        )
        generated_text = output[0].outputs[0].text.strip()
        return generated_text
    except Exception as e:
        return f"Error during generation: {e}"


def process_image(image, text, temperature, max_new_tokens):
    image = resize_image(image, 640)
    start_time = time.time()
    
    if model_type == "transformers":
        generated_text = generate_transformers(processor, model, image, text, temperature, max_new_tokens)
    else:
        generated_text = generate_vllm(processor, model, image, text, temperature, max_new_tokens)
    
    inference_time = time.time() - start_time
    coords = get_coords(generated_text, image)

    if isinstance(coords, str):
        return None, generated_text, inference_time
    
    points = np.array(coords)
    labels = np.ones(len(points), dtype=np.int32)

    if isinstance(image, Image.Image):
        image = np.array(image)
    
    fig = show_points(image, points)
    return fig, generated_text, inference_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose the model to load (transformers or vLLM).")
    parser.add_argument("--use-vllm", action="store_true",
                        help="If specified, uses the vLLM model; otherwise, defaults to transformers.")
    parser.add_argument("--launch-mode", type=str, choices=["local", "share", "network"], default="local",
                        help="Specify launch mode: 'local' (localhost), 'share' (public URL), or 'network' (local network).")
    args = parser.parse_args()

    if args.use_vllm:
        processor, model, model_type = load_vllm_model()
    else:
        processor, model, model_type = load_transformers_model()

    if processor is None or model is None:
        print("Error: Model could not be loaded. Exiting.")
        exit(1)
    
    app = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Upload an Image"),
            gr.Textbox(label='Prompt', placeholder='e.g., point to drink'),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Temperature"),
            gr.Slider(minimum=256, maximum=2048, value=512, label="Max New Tokens")
        ],
        outputs=[
            gr.Plot(label="Image with Points", format="png"),
            gr.Textbox(label="Generated Text"),
            gr.Textbox(label="Inference Time")
        ],
        title="Molmo-7B-D Demo",
        theme="NoCrypt/miku"
    )

    if args.launch_mode == "share":
        app.launch(share=True)
    elif args.launch_mode == "network":
        app.launch(server_name="0.0.0.0")
    else:
        app.launch()
