from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
}


def get_target_layer(model):
    """Last conv block before global average pooling, the usual Grad-CAM hook point."""
    return [model.layer4[-1]]


def compute_saliency(model, image_tensor, target_class, method="gradcam"):
    """Saliency map for one image. image_tensor is (1, C, H, W), normalized.
    Returns an (H, W) array in [0, 1]."""
    cam_class = CAM_METHODS[method]
    target_layers = get_target_layer(model)

    with cam_class(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    return grayscale_cam[0]
