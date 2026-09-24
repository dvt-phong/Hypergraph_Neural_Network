# 7. Define weighted BCE, symmetric contrastive loss, and their sum.

import torch
from torch.nn import functional as F


# Compute the positive-class weight from binary train labels.
def train_pos_weight(labels):
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both classes are required in the train split")
    return negatives / positives


# Compare initial and refined embeddings for the sampled nodes.
def contrastive_loss(
    initial_node_embeddings,
    refined_node_embeddings,
    sampled_node_indices,
    temperature=0.2,
):
    if temperature <= 0 or len(sampled_node_indices) < 2:
        raise ValueError("Invalid contrastive temperature or node sample")
    initial_embeddings = F.normalize(
        initial_node_embeddings[sampled_node_indices],
        dim=1,
    )
    refined_embeddings = F.normalize(
        refined_node_embeddings[sampled_node_indices],
        dim=1,
    )
    similarity_matrix = initial_embeddings @ refined_embeddings.T / temperature
    matching_positions = torch.arange(
        len(sampled_node_indices),
        device=initial_node_embeddings.device,
    )
    initial_to_refined_loss = F.cross_entropy(
        similarity_matrix,
        matching_positions,
    )
    refined_to_initial_loss = F.cross_entropy(
        similarity_matrix.T,
        matching_positions,
    )
    return (initial_to_refined_loss + refined_to_initial_loss) / 2


# Combine weighted classification loss with the optional contrastive component.
def total_loss(
    model_output,
    labels,
    positive_weight,
    sampled_node_indices,
    *,
    lambda_cl=0.1,
    temperature=0.2,
):
    classification_loss = F.binary_cross_entropy_with_logits(
        model_output["logits"],
        labels,
        pos_weight=positive_weight,
    )
    if lambda_cl > 0:
        contrastive_loss_value = contrastive_loss(
            model_output["z0"],
            model_output["z_star"],
            sampled_node_indices,
            temperature,
        )
    else:
        contrastive_loss_value = classification_loss.new_zeros(())

    combined_loss = classification_loss + lambda_cl * contrastive_loss_value
    loss_components = {
        "bce": float(classification_loss.detach()),
        "contrastive": float(contrastive_loss_value.detach()),
    }
    return combined_loss, loss_components
