from argparse import Namespace
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.StyleGANWrapper import StyleGANWrapper
import torch



# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')


def display_alongside_source_image(result_image, source_image, resize_dims=(256, 256)):
    res = np.concatenate(
        [
            np.array(source_image.resize(resize_dims)),
            np.array(result_image.resize(resize_dims)),
        ],
        axis=1,
    )
    return Image.fromarray(res)


def get_file_list(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def is_image_file(file_name):
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".svg",
        ".ico",
    ]
    return any(file_name.lower().endswith(ext) for ext in image_extensions)


def get_path_list(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and is_image_file(f)
    ]


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def load_model(model_path = "./pretrained_models/stylegan_encoder_decoder.pt"):
    ckpt = torch.load(model_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["is_train"] = False
    opts["checkpoint_path"] = model_path
    opts = Namespace(**opts)
    net = StyleGANWrapper(opts)
    net.eval()
    net.cuda()
    return net