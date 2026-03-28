"""
으뜸 딥러닝 — 12장 06절
제로샷 분류의 유사도 히트맵
"""

import numpy as np

# Select sample images (one per class)
sample_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
sim_matrix = []

for idx in sample_indices:
    pil_img = Image.fromarray(cifar_test.data[idx])
    img_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model_clip.encode_image(img_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    sims = (img_feat @ text_features.T).squeeze(0).cpu().numpy()
    sim_matrix.append(sims)

sim_matrix = np.array(sim_matrix)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(10))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(10))
ax.set_yticklabels(
    [f"img {i}" for i in range(10)], fontsize=9
)
ax.set_xlabel("Text prompt")
ax.set_ylabel("Image")
plt.colorbar(im, ax=ax, label="Cosine similarity")
plt.title("CLIP Image-Text Similarity")
plt.tight_layout()
plt.show()
