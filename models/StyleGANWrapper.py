import matplotlib

import models.encoders.ResidualEncoder

matplotlib.use("Agg")
import torch
from torch import nn
from models.encoders import psp_encoders
from models.decoder.model import Generator  # , Discriminator

# from configs.paths_config import model_paths
import torchvision.transforms as transforms


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


class StyleGANWrapper(nn.Module):
    def __init__(self, opts):
        super(StyleGANWrapper, self).__init__()
        self.opts = opts
        self.encoder = self.define_encoder()
        self.residue = models.encoders.ResidualEncoder.ResidualEncoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.grid_transform = transforms.RandomPerspective(
            distortion_scale=opts.distortion_scale, p=opts.aug_rate
        )
        self.grid_align = models.encoders.ResidualEncoder.ResidualAligner()
        self.load_weights()

    def define_encoder(self):
        enc_types = {
            "GradualStyleEncoder": psp_encoders.GradualStyleEncoder,
            "Encoder4Editing": psp_encoders.Encoder4Editing,
        }
        if self.opts.encoder_type in enc_types:
            return enc_types[self.opts.encoder_type](50, "ir_se", self.opts)
        raise Exception(f"{self.opts.encoder_type} is not a valid encoder")

    def load_weights(self):
        if self.opts.checkpoint_path:
            print(f"Loading encoder from: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, "decoder"), strict=True)
            self._load_latent_avg(ckpt)
            if not self.opts.is_train:
                self.residue.load_state_dict(get_keys(ckpt, "residue"), strict=True)
                self.grid_align.load_state_dict(
                    get_keys(ckpt, "grid_align"), strict=True
                )
        else:
            print("Checkpoint path not provided!")

    def forward(
        self,
        x,
        resize=True,
        latent_mask=None,
        input_code=False,
        randomize_noise=True,
        inject_latent=None,
        return_latents=False,
        alpha=None,
    ):
        codes = x if input_code else self.encoder(x)
        codes += self._apply_latent_avg(codes)

        images, result_latent = self._decode(
            codes, input_code, randomize_noise, return_latents
        )
        res_aligned, res_gt = self._align_residuals(x, images)
        conditions = self.residue(res_aligned.to(self.opts.device))
        if conditions is not None:
            images, result_latent = self._decode(
                codes, input_code, randomize_noise, return_latents, conditions
            )

        images = self.face_pool(images) if resize else images
        return (
            (images, result_latent, res_aligned - res_gt, images)
            if return_latents
            else images
        )

    def _load_latent_avg(self, ckpt, repeat=None):
        self.latent_avg = (
            ckpt.get("latent_avg", None).to(self.opts.device)
            if "latent_avg" in ckpt
            else None
        )
        if self.latent_avg is not None and repeat:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)

    def _apply_latent_avg(self, codes):
        if self.opts.start_from_latent_avg and self.latent_avg is not None:
            return self.latent_avg.repeat(codes.shape[0], 1, 1)
        return 0

    def _modify_codes(self, codes, latent_mask, inject_latent, alpha):
        if latent_mask is not None:
            for i in latent_mask:
                codes[:, i] = inject_latent[:, i] if inject_latent is not None else 0
                if alpha is not None and inject_latent is not None:
                    codes[:, i] = (
                        alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    )
        return codes

    def _decode(
        self, codes, input_code, randomize_noise, return_latents, conditions=None
    ):
        input_is_latent = not input_code
        return self.decoder(
            [codes],
            conditions,
            input_is_latent=input_is_latent,
            randomize_noise=randomize_noise,
            return_latents=return_latents,
        )

    def _align_residuals(self, x, images):
        res_gt = (
            x
            - torch.nn.functional.interpolate(
                torch.clamp(images, -1.0, 1.0), size=(256, 256), mode="bilinear"
            )
        ).detach()
        res_unaligned = self.grid_transform(res_gt).detach()
        return self.grid_align(torch.cat((res_unaligned, images), 1)), res_gt
