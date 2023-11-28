import math
import os
import torch
from torch import optim
from tqdm import tqdm

from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from criteria.lpips.lpips import LPIPS

# from models.stylegan2_vanila.model import Generator


class LatentAttack:
    def __init__(self, model, fr_model=["irse50"], victim_model=["cur_face"]):
        self.generator = model.eval().cuda()
        self.lpips_loss = LPIPS()
        self.victim = IDLoss(victim_model)
        self.id_loss = IDLoss(fr_model)

    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    def guided_attack(self, latent_src, latent_tar, args, tar_id=None):
        # print(f" id lambda {args}")
        # Set default argument values

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
            # if verbo:
            # pbar.set_description((f"loss: {loss.item():.4f} --- id_loss: {i_loss:.4f}"))
            id_dist_to_src = self.victim(img_src, img_gen)[0].detach().cpu().numpy()
            # if i % 5:
            #     print(
            #         f"loss: {loss.item():.4f} -- ID loss: {i_loss:.4f}; -- ID eval: {id_dist_to_src:.4f}."
            #     )
            i += 1
            if i > 100:
                # return blank image
                print("Breaking early, not successful")
                return img_gen, img_src, latent, id_dist_to_src
        # id_dist_to_tar = (
        #     self.victim(img_gen, tar_id_img.unsqueeze(0))[0].detach().cpu().numpy()
        # )
        return img_gen, img_src, latent, id_dist_to_src

    def no_guidance_attack(self, args, tar_id=None):
        self.generator.eval()
        mean_latent = self.generator.mean_latent(4096)

        if args.latent is not None:
            latent_code_init = args.latent.cuda()

        with torch.no_grad():
            img_orig, _ = self.generator(
                [latent_code_init],
                input_is_latent=True,
                randomize_noise=False,
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

            c_loss = lpips_loss(img_gen, img_orig)
            if args.id_lambda > 0 and tar_id is None:
                i_loss = self.id_loss(img_gen, img_orig)[0]
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

            eval_id = self.victim(img_gen, img_orig)[0]

            i += 1
            if i > 150:
                print("Breaking early, not successful")
                return img_orig, img_gen, latent

        print(f"total steps: {i}")
        if args.mode == "edit":
            final_result = torch.cat([img_orig, img_gen])
        else:
            final_result = img_gen
        if args.return_latent:
            return img_orig, img_gen, latent
        return final_result
