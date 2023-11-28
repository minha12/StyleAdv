import os

google_drive_paths = {
    "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
    "mapper/pretrained/afro.pt": "https://drive.google.com/uc?id=1i5vAqo4z0I-Yon3FNft_YZOq7ClWayQJ",
    "mapper/pretrained/angry.pt": "https://drive.google.com/uc?id=1g82HEH0jFDrcbCtn3M22gesWKfzWV_ma",
    "mapper/pretrained/beyonce.pt": "https://drive.google.com/uc?id=1KJTc-h02LXs4zqCyo7pzCp0iWeO6T9fz",
    "mapper/pretrained/bobcut.pt": "https://drive.google.com/uc?id=1IvyqjZzKS-vNdq_OhwapAcwrxgLAY8UF",
    "mapper/pretrained/bowlcut.pt": "https://drive.google.com/uc?id=1xwdxI2YCewSt05dEHgkpmmzoauPjEnnZ",
    "mapper/pretrained/curly_hair.pt": "https://drive.google.com/uc?id=1xZ7fFB12Ci6rUbUfaHPpo44xUFzpWQ6M",
    "mapper/pretrained/depp.pt": "https://drive.google.com/uc?id=1FPiJkvFPG_y-bFanxLLP91wUKuy-l3IV",
    "mapper/pretrained/hilary_clinton.pt": "https://drive.google.com/uc?id=1X7U2zj2lt0KFifIsTfOOzVZXqYyCWVll",
    "mapper/pretrained/mohawk.pt": "https://drive.google.com/uc?id=1oMMPc8iQZ7dhyWavZ7VNWLwzf9aX4C09",
    "mapper/pretrained/purple_hair.pt": "https://drive.google.com/uc?id=14H0CGXWxePrrKIYmZnDD2Ccs65EEww75",
    "mapper/pretrained/surprised.pt": "https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI",
    "mapper/pretrained/taylor_swift.pt": "https://drive.google.com/uc?id=10jHuHsKKJxuf3N0vgQbX_SMEQgFHDrZa",
    "mapper/pretrained/trump.pt": "https://drive.google.com/uc?id=14v8D0uzy4tOyfBU3ca9T0AzTt3v-dNyh",
    "mapper/pretrained/zuckerberg.pt": "https://drive.google.com/uc?id=1NjDcMUL8G-pO3i_9N6EPpQNXeMc3Ar1r",
    "example_celebs.pt": "https://drive.google.com/uc?id=1VL3lP4avRhz75LxSza6jgDe-pHd2veQG",
}


def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url,
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights.",
        )


STYLESPACE_DIMENSIONS = (
    [512 for _ in range(15)]
    + [256, 256, 256]
    + [128, 128, 128]
    + [64, 64, 64]
    + [32, 32]
)


def aggregate_loss_dict(agg_loss_dict):
    mean_vals = {}
    for output in agg_loss_dict:
        for key in output:
            mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
    for key in mean_vals:
        if len(mean_vals[key]) > 0:
            mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
        else:
            print("{} has no value".format(key))
            mean_vals[key] = 0
    return mean_vals


def convert_s_tensor_to_list(batch):
    s_list = []
    for i in range(len(STYLESPACE_DIMENSIONS)):
        s_list.append(batch[:, :, 512 * i : 512 * i + STYLESPACE_DIMENSIONS[i]])
    return s_list


def padding_s_latent(latents):
    latents = [torch.tensor(code).unsqueeze(-1).unsqueeze(-1) for code in latents]
    padded_latents = []
    for latent in latents:
        latent = latent.cpu()
        if latent.shape[2] == 512:
            padded_latents.append(latent)
        else:
            padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
            padded_latent = torch.cat([latent, padding], dim=2)
            padded_latents.append(padded_latent)
    latents = torch.cat(padded_latents, dim=2)
    return latents


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    result = Image.fromarray(var.astype("uint8"))
    return result
