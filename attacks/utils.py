from PIL import Image


def swap_layers(original_tensor, reference_tensor, layers_to_swap):
    # Start with a copy of the original tensor
    modified_tensor = original_tensor.clone()

    # Replace specified layers in the modified tensor with those from the reference tensor
    for layer in layers_to_swap:
        modified_tensor[:, layer, :] = reference_tensor[:, layer, :]

    return modified_tensor


def save_image(tensor, filename):
    tensor = tensor.squeeze() * 0.5 + 0.5  # Remove the batch dimension
    tensor = tensor.clamp(0, 1)  # Clamp values to be within [0, 1]
    tensor = tensor.mul(255).byte()  # Convert to uint8
    tensor = (
        tensor.cpu().numpy().transpose((1, 2, 0))
    )  # Convert to numpy array in HxWxC format

    # Convert to a PIL.Image and save
    image = Image.fromarray(tensor)
    image.save(filename)
