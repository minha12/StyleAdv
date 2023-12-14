import sys
from attacks.latent_attack import LatentAttack
from argparse import Namespace
from attacks.residual_attack import residual_attack
from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from datasets.images_dataset import ImageDataset
from editings import latent_editor
from attacks.utils import swap_layers, save_image
from utils.align import process_input_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.common import (
    display_alongside_source_image,
    get_latents,
    tensor2im,
)


import torch
from torchvision.utils import save_image


import os

from utils.model_utils import load_model

net = load_model()
editor = latent_editor.LatentEditor(net.decoder)
interfacegan_directions = {
    "age": "./editings/interfacegan_directions/age.pt",
    "smile": "./editings/interfacegan_directions/smile.pt",
}
ganspace_pca = torch.load("./editings/ganspace_pca/ffhq_pca.pt")
ganspace_directions = {
    "eyes": (54, 7, 8, 20),
    "beard": (58, 7, 9, -20),
    "lip": (34, 10, 11, 20),
}


def launch_guided_attack(
    ckpt_input,
    target_image,
    latent_codes,
    edit_latents,
    stylegan_size,
    lr_rampup,
    lr,
    id_threshold,
    mode,
    l2_lambda,
    id_lambda,
    truncation,
    id_loss_model,
    id_eval_model,
    seed_value,
    return_latent,
    batch_size,
    orig,
):
    model = net
    attacker = LatentAttack(model, id_loss_model, id_eval_model)
    args = Namespace(
        stylegan_size=stylegan_size,
        lr_rampup=lr_rampup,
        lr=lr,
        id_threshold=id_threshold,
        mode=mode,
        lpips_lambda=1.0,
        l2_lambda=l2_lambda,
        id_lambda=id_lambda,
        latent_path=None,
        truncation=truncation,
        save_intermediate_image_every=0,
        results_dir="results",
        return_latent=return_latent,
        batch_size=batch_size,
    )
    img_gen, _, latent, _ = attacker.guided_attack(
        latent_codes, edit_latents, orig, args, tar_id=target_image
    )
    return img_gen, latent


# untargeted attack
def launch_no_guidance_attack(
    ckpt_input,
    latent,
    stylegan_size,
    lr_rampup,
    lr,
    id_threshold,
    mode,
    l2_lambda,
    id_lambda,
    truncation,
    id_loss_model,
    id_eval_model,
    seed_value,
    return_latent,
    batch_size,
    orig,
):
    model = net
    attacker = LatentAttack(model, id_loss_model, id_eval_model)
    args = Namespace(
        model=model,
        latent=latent,
        stylegan_size=stylegan_size,
        lr_rampup=lr_rampup,
        lr=lr,
        id_threshold=id_threshold,
        mode=mode,
        l2_lambda=l2_lambda,
        id_lambda=id_lambda,
        latent_path=None,
        truncation=truncation,
        save_intermediate_image_every=0,
        results_dir="results",
        id_loss_model=id_loss_model,
        id_eval_model=id_eval_model,
        return_latent=return_latent,
        batch_size=batch_size,
    )
    if seed_value != -1:
        torch.manual_seed(seed_value)
    img_orig, img_gen, latent = attacker.no_guidance_attack(orig, args, tar_id=None)

    return img_gen, latent


def styleadv(
    input_path,
    output_path,
    target_image_path,
    app_mode,
    editing_direction,
    edit_degree,
    lr,
    id_threshold,
    l2_lambda,
    id_lambda,
    id_loss_model,
    id_eval_model,
    batch_size=1,
    ckpt_input="",
    stylegan_size=1024,
    lr_rampup=0.05,
    mode="edit",
    truncation=0.7,
    seed_value=-1,
    return_latent=True,
    # additional parameters
    demo=True,
    target_id_attack=False,
    align_image=True,
    num_workers=1,
):
    if demo:
        # Align the input image
        input_image = process_input_image(input_path)
        x = input_image.unsqueeze(0).cuda()
        latent_codes = get_latents(net, x)
        new_latent_codes = latent_codes.clone()

        result = actions(
            target_image_path,
            app_mode,
            editing_direction,
            edit_degree,
            lr,
            id_threshold,
            l2_lambda,
            id_lambda,
            id_loss_model,
            id_eval_model,
            batch_size,
            ckpt_input,
            stylegan_size,
            lr_rampup,
            mode,
            truncation,
            seed_value,
            return_latent,
            x,
            latent_codes,
            new_latent_codes,
        )
        result = torch.nn.functional.interpolate(
            result, size=(256, 256), mode="bilinear"
        )
        # evaluator = IDLoss(id_eval_model)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        tensor2im(result[0]).save(
            os.path.join(output_path, os.path.basename(input_path))
        )
        return display_alongside_source_image(
            tensor2im(result[0]), tensor2im(input_image)
        )
    else:
        # Create the custom dataset
        custom_dataset = ImageDataset(
            input_path, return_relative_paths=True, run_align=align_image
        )

        # Create the DataLoader
        data_loader = DataLoader(
            custom_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        # evaluator = IDLoss(id_eval_model).eval().cuda()
        print(f"Loanding total {len(data_loader)} images from dataset")
        os.makedirs(output_path, exist_ok=True)
        for batch, filenames in tqdm(data_loader):
            x = batch.cuda()
            latent_codes = get_latents(net, x)
            new_latent_codes = latent_codes.clone()

            result = actions(
                target_image_path,
                app_mode,
                editing_direction,
                edit_degree,
                lr,
                id_threshold,
                l2_lambda,
                id_lambda,
                id_loss_model,
                id_eval_model,
                batch_size,
                ckpt_input,
                stylegan_size,
                lr_rampup,
                mode,
                truncation,
                seed_value,
                return_latent,
                x,
                latent_codes,
                new_latent_codes,
            )
            result = torch.nn.functional.interpolate(
                result, size=(512, 512), mode="bilinear"
            )
            for res, f_name in zip(result, filenames):
                result_image_output = tensor2im(res)
                path = os.path.join(output_path, f_name)
                result_image_output.save(path, format="PNG", bit=16)


def actions(
    target_image_path,
    app_mode,
    editing_direction,
    edit_degree,
    lr,
    id_threshold,
    l2_lambda,
    id_lambda,
    id_loss_model,
    id_eval_model,
    batch_size,
    ckpt_input,
    stylegan_size,
    lr_rampup,
    mode,
    truncation,
    seed_value,
    return_latent,
    x,
    latent_codes,
    new_latent_codes,
):
    if app_mode == "Residual Attack":
        with torch.no_grad():
            imgs, _ = net.decoder(
                [new_latent_codes.cuda()],
                None,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
            )
            diff = x - torch.nn.functional.interpolate(
                torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
            )
        result, _ = residual_attack(
            net,
            x,
            imgs,
            new_latent_codes,
            diff,
            id_loss_model,
            id_eval_model,
            lr=lr,
            l2_lambda=l2_lambda,
            id_threshold=id_threshold,
        )

    elif app_mode == "Guided Adversarial Editing":
        if target_image_path != "":
            target_image = process_input_image(target_image_path).unsqueeze(0).cuda()
        else:
            target_image = None

        if editing_direction in interfacegan_directions:
            edit_direction = torch.load(
                interfacegan_directions[editing_direction]
            ).cuda()
            img_edit, edit_latents = editor.apply_interfacegan(
                latent_codes.cuda(),
                edit_direction,
                factor=edit_degree,
            )
        elif editing_direction in ganspace_directions:
            edit_direction = ganspace_directions[editing_direction]
            img_edit, edit_latents = editor.apply_ganspace(
                latent_codes.cuda(), ganspace_pca, [edit_direction]
            )
        # apply targeted adversarial attack
        result, edit_latents = launch_guided_attack(
            ckpt_input,
            target_image,
            latent_codes,
            edit_latents,
            stylegan_size,
            lr_rampup,
            lr,
            id_threshold,
            mode,
            l2_lambda,
            id_lambda,
            truncation,
            id_loss_model,
            id_eval_model,
            seed_value,
            return_latent,
            batch_size,
            orig=x,
        )
    elif app_mode == "No Guidance Adversarial Editing":
        result, _ = launch_no_guidance_attack(
            ckpt_input,
            latent_codes.detach().cpu(),
            stylegan_size,
            lr_rampup,
            lr,
            id_threshold,
            mode,
            l2_lambda,
            id_lambda,
            truncation,
            id_loss_model,
            id_eval_model,
            seed_value,
            return_latent,
            batch_size,
            orig=x,
        )
    return result
