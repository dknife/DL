"""
으뜸 딥러닝 — 15장 01절
지식 증류 손실 함수
"""

import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits,
                      labels, T=4.0, alpha=0.7):
    # Soft loss: KL divergence between soft outputs
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    loss_soft = F.kl_div(soft_student, soft_teacher,
                         reduction="batchmean")

    # Hard loss: standard cross-entropy with true labels
    loss_hard = F.cross_entropy(student_logits, labels)

    return alpha * T * T * loss_soft + (1 - alpha) * loss_hard
