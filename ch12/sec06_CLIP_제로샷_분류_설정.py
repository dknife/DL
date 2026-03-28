"""
으뜸 딥러닝 — 12장 06절
CLIP 제로샷 분류 설정
"""

import clip
from PIL import Image

# Load pretrained CLIP model
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# Define class prompts for CIFAR-10
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
prompts = [f"a photo of a {c}" for c in classes]
text_tokens = clip.tokenize(prompts).to(device)

# Encode all text prompts
with torch.no_grad():
    text_features = model_clip.encode_text(text_tokens)
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )
