---
license: apache-2.0
datasets:
- allenai/Molmo2-Cap
- allenai/Molmo2-VideoCapQA
- allenai/Molmo2-VideoSubtitleQA
- allenai/Molmo2-AskModelAnything
- allenai/Molmo2-VideoPoint
- allenai/Molmo2-VideoTrack
- allenai/Molmo2-MultiImageQA
- allenai/Molmo2-SynMultiImageQA
- allenai/Molmo2-MultiImagePoint
language:
- en
base_model:
- google/siglip-so400m-patch14-384
- Qwen/Qwen3-4B-Instruct-2507
pipeline_tag: video-text-to-text
library_name: transformers
tags:
- multimodal
- olmo
- molmo
- molmo2
---

<img src="molmo_2_logo_RGB.png" alt="Logo for the Molmo2 Project" style="width: auto; height: 50px;">

# Molmo2-4B

Molmo2 is a family of open vision-language models developed by the Allen Institute for AI (Ai2) that support image, video and multi-image understanding and grounding.
Molmo2 models are trained on publicly available third party datasets as referenced in [our technical report](https://allenai.org/papers/molmo2) and [Molmo2 data](https://huggingface.co/collections/allenai/molmo2-data), 
a collection of datasets with highly-curated image-text and video-text pairs.
It has state-of-the-art performance among multimodal models with a similar size.
You can find all models in the Molmo2 family [here](https://huggingface.co/collections/allenai/molmo2).

**Learn more** about the Molmo2 family [in our announcement blog post](https://allenai.org/blog/molmo2).

Molmo2-4B is based on [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and uses [SigLIP 2](https://huggingface.co/google/siglip-so400m-patch14-384) as vision backbone.
It outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos.

Ai2 is commited to open science. The Molmo2 datasets are available [here](https://huggingface.co/collections/allenai/molmo2-data). 
All other artifacts used in creating Molmo2 (training code, evaluations, intermediate checkpoints) will be made available at a later date, furthering our commitment to open-source AI development and reproducibility.

Quick links:
- 📂 [All Models](https://huggingface.co/collections/allenai/molmo2)
- 📃 [Paper](https://allenai.org/papers/molmo2)
- 🎥 [Blog with Videos](https://allenai.org/blog/molmo2)

## Quick Start

### Setup Conda Environment
```
conda create --name transformers4571 python=3.11
conda activate transformers4571
pip install transformers==4.57.1
pip install torch pillow einops torchvision accelerate decord2 molmo_utils
```

### General Video QA

```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
     model_id,
     trust_remote_code=True,
     dtype="auto",
     device_map="auto"
)

# process the video and text
messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Which animal appears in the video?"),
            dict(type="video", video="https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"),
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)
```

### Pointing Video QA

```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from molmo_utils import process_vision_info
import re

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    all_points = []
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_video_points(text, image_w, image_h, extract_ids=False):
    """Extract video pointing coordinates as a flattened list of (t, x, y) triplets from model output text."""
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))
            w, h = (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Point to the penguins."),
            dict(type="video", video="https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"),
        ],
    }
]

# process the video using `molmo_utils.process_vision_info`
_, videos, video_kwargs = process_vision_info(messages)
videos, video_metadatas = zip(*videos)
videos, video_metadatas = list(videos), list(video_metadatas)

# apply the chat template to the input messages
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# process the video and text
inputs = processor(
    videos=videos,
    video_metadata=video_metadatas,
    text=text,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# decode video pointing outputs
points = extract_video_points(generated_text, image_w=video_metadatas[0]["width"], image_h=video_metadatas[0]["height"])
print(points)
```

### Tracking Video QA

```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from molmo_utils import process_vision_info
import re

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    all_points = []
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_video_points(text, image_w, image_h, extract_ids=False):
    """Extract video pointing coordinates as a flattened list of (t, x, y) triplets from model output text."""
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))
            w, h = (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Track the player who is dunking"),
            dict(type="video", video="https://storage.googleapis.com/oe-training-public/demo_videos/arena_basketball.mp4"),
        ],
    }
]

# process the video using `molmo_utils.process_vision_info`
_, videos, video_kwargs = process_vision_info(messages)
videos, video_metadatas = zip(*videos)
videos, video_metadatas = list(videos), list(video_metadatas)

# apply the chat template to the input messages
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# process the video and text
inputs = processor(
    videos=videos,
    video_metadata=video_metadatas,
    text=text,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# decode video pointing outputs
points = extract_video_points(generated_text, image_w=video_metadatas[0]["width"], image_h=video_metadatas[0]["height"])
print(points)
```

### Multi-image QA
```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import requests
from PIL import Image

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)

# process the image and text
messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Compare these images."),
            dict(type="image", image=Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)),
            dict(type="image", image=Image.open(requests.get("https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg", stream=True).raw))
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=448)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)
```

### Multi-Image Point QA

```
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import re
from PIL import Image
import requests

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
    token=True
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
    token=True
)

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    all_points = []
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """Extract pointing coordinates as a flattened list of (frame_id, x, y) triplets from model output text."""
    all_points = []
    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        diff_res = True
    else:
        diff_res = False
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
            w, h = (image_w[frame_id-1], image_h[frame_id-1]) if diff_res else (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

# process the image and text
images = [
    Image.open(requests.get("https://storage.googleapis.com/oe-training-public/demo_images/boat1.jpeg", stream=True).raw),
    Image.open(requests.get("https://storage.googleapis.com/oe-training-public/demo_images/boat2.jpeg", stream=True).raw)
]

messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Point to the boats"),
            dict(type="image", image=images[0]),
            dict(type="image", image=images[1]),
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

points = extract_multi_image_points(
    generated_text,
    [images[0].width, images[1].width],
    [images[0].height, images[1].height],
)
print(points)
```

## Evaluations

We report the Average Score on 15 Academic Benchmarks here.
For details on the evals, refer to the main video results table in our [technical report](https://allenai.org/papers/molmo2).

| Model | Average Score on 15 Academic Benchmarks |
|-----------------------------|-----------------------------------------|
| GPT-5 | 70.6 |
| GPT-5 mini | 65.0 |
| Gemini 3 Pro | 70.0 |
| Gemini 2.5 Pro | 71.2 |
| Gemini 2.5 Flash | 66.7 |
| Claude Sonnet 4.5 | 59.6 |
| InternVL3.5-4B | 53.4 |
| InternVL3.5-8B | 54.1 |
| Qwen3-VL-4B | 58.1 |
| Qwen3-VL-8B | 59.5 |
| Keye-VL-1.5-8B | 55.7 |
| GLM-4.1V-9B | 56.9 |
| MiniCPM-V-4.5-8B | 56.6 |
| Eagle2.5-8B | 60.7 |
| PLM-3B | 53.9 |
| PLM-8B | 56.2 |
| LLaVA-Video-7B | 52.7 |
| VideoChat-Flash-7B | 56.1 |
| **Molmo2-4B (this model)** | 62.8 |
| Molmo2-8B | 63.1 |
| Molmo2-7B | 59.7 |

## License and Use

This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2’s [Responsible Use Guidelines](https://allenai.org/responsible-use).
This model is trained on third party datasets that are subject to academic and non-commercial research use only. Please review the sources to determine if this model is appropriate for your use case.