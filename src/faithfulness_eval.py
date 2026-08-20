import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gradcam_utils import compute_saliency

PERCENTAGES = [5, 10, 20, 30, 50]
CAM_METHODS = ["gradcam", "gradcam++", "hirescam"]
MASK_VALUE = 0.0  # 0 after normalization is the same as the ImageNet mean pixel


def get_targeted_mask(saliency_map, percent):
    """(H, W) boolean mask, True for the top `percent`% most salient pixels."""
    threshold = np.percentile(saliency_map, 100 - percent)
    return saliency_map >= threshold


def get_random_mask(n_true, shape, rng):
    """Same number of True pixels as get_targeted_mask, but picked at random.
    This is the control condition."""
    mask = np.zeros(shape, dtype=bool)
    flat_indices = rng.choice(mask.size, size=n_true, replace=False)
    mask.flat[flat_indices] = True
    return mask


def apply_mask(image_tensor, spatial_mask, mask_value=MASK_VALUE):
    """image_tensor is (1, C, H, W). spatial_mask is (H, W), True = masked out."""
    mask = torch.from_numpy(spatial_mask).to(image_tensor.device)
    masked = image_tensor.clone()
    masked[:, :, mask] = mask_value
    return masked


@torch.no_grad()
def predict(model, image_tensor, target_class):
    """Returns (predicted_class, confidence in target_class) for one image."""
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(1).item()
    target_confidence = probs[0, target_class].item()
    return pred_class, target_confidence


def run_deletion_curve(model, image_tensor, true_label, cam_method, device, rng,
                        percentages=PERCENTAGES):
    """Runs targeted vs random masking on one image, for one CAM method.
    Returns one result row per (percentage, strategy) pair."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        orig_pred = probs.argmax(1).item()
        orig_conf = probs[0, orig_pred].item()

    saliency_map = compute_saliency(model, image_tensor, target_class=orig_pred, method=cam_method)

    rows = []
    for percent in percentages:
        targeted_mask = get_targeted_mask(saliency_map, percent)
        n_masked = int(targeted_mask.sum())
        random_mask = get_random_mask(n_masked, saliency_map.shape, rng)

        for strategy, mask in [("targeted", targeted_mask), ("random", random_mask)]:
            masked_image = apply_mask(image_tensor, mask)
            pred_class, conf = predict(model, masked_image, target_class=orig_pred)
            rows.append({
                "cam_method": cam_method,
                "percent": percent,
                "strategy": strategy,
                "true_label": true_label,
                "orig_pred": orig_pred,
                "orig_conf": orig_conf,
                "masked_pred": pred_class,
                "masked_conf": conf,
                "masked_correct": int(pred_class == true_label),
            })

    return rows


def run_full_evaluation(model, test_loader, device, seed=0, cam_methods=CAM_METHODS,
                         percentages=PERCENTAGES):
    """Runs the deletion protocol over the whole test set, for every CAM method.
    Returns a DataFrame with one row per (image, cam_method, percent, strategy)."""
    model.eval()
    rng = np.random.default_rng(seed)

    all_rows = []
    image_id = 0
    for images, labels in test_loader:
        for i in range(images.size(0)):
            image_tensor = images[i:i + 1]
            true_label = labels[i].item() if labels.dim() > 0 else labels.item()

            for cam_method in cam_methods:
                rows = run_deletion_curve(model, image_tensor, true_label, cam_method,
                                           device, rng, percentages)
                for row in rows:
                    row["image_id"] = image_id
                all_rows.extend(rows)

            image_id += 1

    return pd.DataFrame(all_rows)
