import numpy as np
import torch
import torch.nn.functional as F

ss_num = 2
def compute_el2n_score(model, dataloader, ss, device='cuda:2'):
    """Compute EL2N (Error L2 Norm) scores for dataset samples."""
    model.eval()
    el2n_scores = []

    for _, (_, imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        if ss:
            outputs = outputs[:, ::ss_num]

        probs = F.softmax(outputs, dim=1)
        one_hot_labels = F.one_hot(labels, num_classes=outputs.size(1)).float()

        el2n = torch.sum((probs - one_hot_labels)  **  2, dim = 1).cpu().numpy()
        el2n_scores.extend(el2n)

    return np.array(el2n_scores)


def prune_dataset(dataset, el2n_scores, prune_fraction):
    """Prune dataset by removing samples with highest EL2N scores."""
    class_indices = {}
    class_scores = {}

    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        if label not in class_indices:
            class_indices[label] = []
            class_scores[label] = []
        class_indices[label].append(idx)
        class_scores[label].append(el2n_scores[idx])

    keep_indices = []
    for label in class_indices:
        sorted_indices = np.argsort(class_scores[label])[::-1]
        num_to_keep = int(len(class_scores[label]) * (1 - prune_fraction))
        keep_indices.extend([class_indices[label][i] for i in sorted_indices[:num_to_keep]])

    pruned_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return pruned_dataset