from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
}


def get_target_layer(model):
    """Last conv block before global average pooling — standard Grad-CAM hook point."""
    return [model.layer4[-1]]


def compute_saliency(model, image_tensor, target_class, method="gradcam"):
    """Saliency map for a single image.

    image_tensor: (1, C, H, W) tensor, normalized as the model expects.
    target_class: class index to explain (e.g. the model's predicted class).
    Returns a (H, W) numpy array in [0, 1].
    """
    cam_class = CAM_METHODS[method]
    target_layers = get_target_layer(model)

    with cam_class(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    return grayscale_cam[0]
