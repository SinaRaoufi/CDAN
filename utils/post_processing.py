import torch


def enhance_contrast(images, contrast_factor=1.5):
    """
    Enhance the contrast of the input images.
    
    Args:
        images (torch.Tensor): Input images of shape (B, C, H, W).
        contrast_factor (float): Factor to adjust contrast. >1 increases contrast, <1 decreases contrast.
    
    Returns:
        torch.Tensor: Contrast-enhanced images.
    """
    # Normalize the images to [0, 1] if not already
    if images.max() > 1.0:
        images = images / 255.0
    
    # Compute mean intensity for each image in the batch
    mean_intensity = images.mean(dim=(2, 3), keepdim=True)
    
    # Adjust contrast
    enhanced_images = (images - mean_intensity) * contrast_factor + mean_intensity
    
    # Clip values to [0, 1] to ensure valid pixel range
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    
    return enhanced_images

def enhance_color(images, saturation_factor=1.5):
    """
    Enhance the color (saturation) of the input images.
    
    Args:
        images (torch.Tensor): Input images of shape (B, C, H, W).
        saturation_factor (float): Factor to adjust saturation. >1 increases saturation, <1 decreases saturation.
    
    Returns:
        torch.Tensor: Color-enhanced images.
    """
    # Normalize the images to [0, 1] if not already
    if images.max() > 1.0:
        images = images / 255.0
    
    # Convert RGB images to grayscale
    grayscale = 0.2989 * images[:, 0, :, :] + 0.5870 * images[:, 1, :, :] + 0.1140 * images[:, 2, :, :]
    grayscale = grayscale.unsqueeze(1)  # Add channel dimension
    
    # Adjust saturation
    enhanced_images = grayscale + saturation_factor * (images - grayscale)
    
    # Clip values to [0, 1] to ensure valid pixel range
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    
    return enhanced_images