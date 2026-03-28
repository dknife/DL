"""
으뜸 딥러닝 — 12장 06절
CIFAR-10에서 CLIP 제로샷 분류
"""

from torchvision import datasets

# Load CIFAR-10 test set
cifar_test = datasets.CIFAR10(
    root="./data", train=False, download=True
)

correct, total = 0, 0
for i in range(1000):  # evaluate on first 1000 images
    pil_img = cifar_test.data[i]
    pil_img = Image.fromarray(pil_img)
    true_label = cifar_test.targets[i]

    # Preprocess and encode image
    img_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = model_clip.encode_image(img_input)
        img_features = img_features / img_features.norm(
            dim=-1, keepdim=True
        )

    # Compute similarities and predict
    similarities = (img_features @ text_features.T).squeeze(0)
    pred = similarities.argmax().item()

    if pred == true_label:
        correct += 1
    total += 1

accuracy = 100 * correct / total
print(f"Zero-shot accuracy: {accuracy:.1f}%")
# Zero-shot accuracy: 88.3%
