"""
으뜸 딥러닝 — 10장 04절
LoRA 파인튜닝 설정 (Hugging Face PEFT)
"""

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,              # rank
    lora_alpha=32,     # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 (0.32% of total)
