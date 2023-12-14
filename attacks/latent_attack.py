import math
import os
import torch
from torch import optim
from tqdm import tqdm

from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from criteria.lpips.lpips import LPIPS


class LatentAttack:
    def __init__(self, model, fr_model=["irse50"], victim_model=["cur_face"]):
        self.model = model
        self.generator = model.decoder.eval().cuda()
        self.lpips_loss = LPIPS()
        self.victim = IDLoss(victim_model)
        self.id_loss = IDLoss(fr_model)

    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    def guided_attack(self, latent_src, latent_tar, orig, args, tar_id=None):
        # Send latent to device
        latent_src = latent_src.detach().clone().cuda()
        latent_tar = latent_tar.detach().clone().cuda()
        os.makedirs(args.results_dir, exist_ok=True)

        with torch.no_grad():
            img_src, _ = self.generator(
                [latent_src],
                input_is_latent=True,
                randomize_noise=False,
            )
            img_tar, _ = self.generator(
                [latent_tar],
                input_is_latent=True,
                randomize_noise=False,
            )

            # calculate the distortion map

            diff = orig - torch.nn.functional.interpolate(
                torch.clamp(img_src, -1.0, 1.0), size=(256, 256), mode="bilinear"
            )

        img_src = img_src.detach()
        img_tar = img_tar.detach()

        # Init beta
        beta = torch.zeros_like(latent_src).cuda()
        beta.requires_grad = True

        optimizer = optim.Adam([beta], lr=args.lr)

        # pbar = range(args.step)
        i = 0
        id_dist_to_src = 0
        while id_dist_to_src < args.id_threshold:
            t = i / 100
            lr = self.get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            latent = latent_src * beta + (1 - beta) * latent_tar
            img_gen, _ = self.generator(
                [latent],
                input_is_latent=True,
                randomize_noise=False,
            )

            if args.lpips_lambda > 0:
                loss_lpips = self.lpips_loss(img_gen, img_tar)
            else:
                loss_lpips = 0

            if args.id_lambda > 0 and tar_id is None:
                i_loss = self.id_loss(img_gen, img_src)[0]
            elif args.id_lambda > 0 and tar_id is not None:
                i_loss = self.id_loss(img_gen, tar_id)[0]
            else:
                i_loss = 0

            if args.l2_lambda > 0:
                l2_loss = ((latent_tar - latent) ** 2).sum()
            else:
                l2_loss = 0

            if tar_id is None:
                loss = (
                    args.lpips_lambda * loss_lpips
                    + args.l2_lambda * l2_loss
                    - args.id_lambda * i_loss
                )
            else:
                loss = (
                    args.lpips_lambda * loss_lpips
                    + args.l2_lambda * l2_loss
                    + args.id_lambda * i_loss
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Align the distortion map
                img_gen = torch.nn.functional.interpolate(
                    torch.clamp(img_gen, -1.0, 1.0), size=(256, 256), mode="bilinear"
                )
                diff_aligned = self.model.grid_align(torch.cat((diff, img_gen), 1))

                # Fusion
                conditions = self.model.residue(diff_aligned)
                img_gen, _ = self.model.decoder(
                    [latent],
                    conditions,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=True,
                )
                id_dist_to_src = self.victim(orig, img_gen)[0].detach().cpu().numpy()
            i += 1
            if i > 150:
                return img_gen, img_src, latent, id_dist_to_src

        # print(f"SUCCESS")
        return img_gen, img_src, latent, id_dist_to_src

    def no_guidance_attack(self, orig, args, tar_id=None):
        self.generator.eval()
        mean_latent = self.generator.mean_latent(4096)

        if args.latent is not None:
            latent_code_init = args.latent.cuda()

        with torch.no_grad():
            img_src, _ = self.generator(
                [latent_code_init],
                input_is_latent=True,
                randomize_noise=False,
            )
            # calculate the distortion map

            diff = orig - torch.nn.functional.interpolate(
                torch.clamp(img_src, -1.0, 1.0), size=(256, 256), mode="bilinear"
            )

        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

        lpips_loss = LPIPS().cuda()
        optimizer = optim.Adam([latent], lr=args.lr)

        i_loss = 0
        i = 0
        eval_id = 0
        while eval_id < args.id_threshold:
            t = i / 100
            lr = self.get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            img_gen, _ = self.generator(
                [latent],
                input_is_latent=True,
                randomize_noise=False,
            )

            c_loss = lpips_loss(img_gen, img_src)
            if args.id_lambda > 0 and tar_id is None:
                i_loss = self.id_loss(img_gen, img_src)[0]
            elif args.id_lambda > 0 and tar_id is not None:
                i_loss = self.id_loss(img_gen, tar_id)[0]
            else:
                i_loss = 0

            l2_loss = ((latent_code_init - latent) ** 2).sum()
            if tar_id is None:
                loss = c_loss + args.l2_lambda * l2_loss - args.id_lambda * i_loss
            else:
                loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Align the distortion map
                img_gen = torch.nn.functional.interpolate(
                    torch.clamp(img_gen, -1.0, 1.0), size=(256, 256), mode="bilinear"
                )
                diff_aligned = self.model.grid_align(torch.cat((diff, img_gen), 1))

                # Fusion
                conditions = self.model.residue(diff_aligned)
                img_gen, _ = self.model.decoder(
                    [latent],
                    conditions,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=True,
                )
                eval_id = self.victim(img_gen, img_src)[0]

            i += 1
            if i > 150:
                return img_src, img_gen, latent

        # print(f"SUCCESS")
        if args.mode == "edit":
            final_result = torch.cat([img_src, img_gen])
        else:
            final_result = img_gen
        if args.return_latent:
            return img_src, img_gen, latent
        return final_result
