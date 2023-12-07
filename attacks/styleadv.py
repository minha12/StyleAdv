import sys
from attacks.latent_attack import LatentAttack
from argparse import Namespace
from attacks.residual_attack import residual_attack
from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from editings import latent_editor
from attacks.utils import swap_layers, save_image
from utils.align import process_input_image

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
):
    # model = load_model(stylegan_size, ckpt_input)
    model = net.decoder
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
        latent_codes, edit_latents, args, tar_id=target_image
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
):
    # model = load_model(stylegan_size, ckpt_input)
    model = net.decoder
    attacker = LatentAttack(model, id_loss_model, id_eval_model)
    # print(ckpt_input)
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
        # work_in_stylespace=work_in_stylespace,
        save_intermediate_image_every=0,
        results_dir="results",
        id_loss_model=id_loss_model,
        id_eval_model=id_eval_model,
        return_latent=return_latent,
        batch_size=batch_size,
    )
    if seed_value != -1:
        torch.manual_seed(seed_value)
    # print(args)
    img_orig, img_gen, latent = attacker.no_guidance_attack(args)
    # img_orig, img_gen, latent = attack(args)

    return img_gen, latent


def styleadv(
    image_path,
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
    n_batch=1,
    batch_size=1,
    outer_loop=1,
    inner_loop=1,
    use_condition=True,
    layer_choices=range(0, 18),
    # for stylegan generator
    ckpt_input="",
    stylegan_size=1024,
    lr_rampup=0.05,
    mode="edit",
    truncation=0.7,
    seed_value=-1,
    return_latent=True,
):
    # Align the input image

    input_image = process_input_image(image_path)
    print(outer_loop)
    # Inference
    # with torch.no_grad():
    x = input_image.unsqueeze(0).cuda()
    latent_codes = get_latents(net, x)
    new_latent_codes = latent_codes.clone()
    if app_mode == "Residual Attack":
        attack_residual = True
        # calculate the distortion map
        for i in range(outer_loop):
            with torch.no_grad():
                imgs, _ = net.decoder(
                    [new_latent_codes[0].unsqueeze(0).cuda()],
                    None,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=True,
                )
                for step in range(inner_loop - 1):
                    imgs = torch.nn.functional.interpolate(
                        torch.clamp(imgs, -1.0, 1.0),
                        size=(256, 256),
                        mode="bilinear",
                        # align_corners=False,
                    )
                    print(imgs.shape)
                    new_latent_codes = get_latents(net, imgs)
                    new_latent_codes = swap_layers(
                        new_latent_codes,
                        latent_codes,
                        layer_choices,
                    )
                    print(new_latent_codes.shape)
                    imgs, _ = net.decoder(
                        [new_latent_codes.cuda()],
                        None,
                        input_is_latent=True,
                        randomize_noise=False,
                        return_latents=True,
                    )
                if use_condition:
                    res = x - torch.nn.functional.interpolate(
                        torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
                    )
                    save_image(res, f"{output_path}/residual.png")
                    # ADA
                    img_edit = torch.nn.functional.interpolate(
                        torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
                    )
                    res_align = net.grid_align(torch.cat((res, img_edit), 1))
                    save_image(res_align, f"{output_path}/res_align.png")
                    # res_align = img_edit
                    print(f"res_align shape: {res_align.shape}")
                    # consultation fusion
                    conditions = net.residue(res_align)
                    print(
                        f"conditions shape: {len(conditions)} and {[condition.shape for condition in conditions]}"
                    )
                else:
                    conditions = None
            # attack_residual = True
            if attack_residual:
                output, perturbed_res = residual_attack(
                    net,
                    x,
                    imgs,
                    new_latent_codes,
                    res,
                    id_loss_model,
                    id_eval_model,
                    lr=lr,
                    l2_lambda=l2_lambda,
                    id_threshold=id_threshold,
                )
                result = output
            else:
                result, _ = net.decoder(
                    [new_latent_codes],
                    conditions,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=True,
                )

                imgs = torch.nn.functional.interpolate(
                    torch.clamp(result, -1.0, 1.0),
                    size=(256, 256),
                    mode="bilinear",
                    # align_corners=False,
                )
                # result = imgs + res
                new_latent_codes = get_latents(net, imgs)
                # imgs = result
                result = imgs
                save_image(result, f"{output_path}/inverted.png")
        # print(result.shape)

    elif app_mode == "Guided Adversarial Editing":
        if target_image_path != "":
            target_image = process_input_image(target_image_path).unsqueeze(0).cuda()
        else:
            target_image = None
        # calculate the distortion map
        imgs, _ = net.decoder(
            [latent_codes[0].unsqueeze(0).cuda()],
            None,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
        for step in range(inner_loop - 1):
            imgs = torch.nn.functional.interpolate(
                torch.clamp(imgs, -1.0, 1.0),
                size=(256, 256),
                mode="bilinear",
                # align_corners=False,
            )
            print(imgs.shape)
            latent_codes = get_latents(net, imgs)
            print(latent_codes.shape)
            imgs, _ = net.decoder(
                [latent_codes.cuda()],
                None,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
            )
        res = x - torch.nn.functional.interpolate(
            torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )

        if editing_direction in interfacegan_directions:
            edit_direction = torch.load(
                interfacegan_directions[editing_direction]
            ).cuda()
            img_edit, edit_latents = editor.apply_interfacegan(
                latent_codes[0].unsqueeze(0).cuda(),
                edit_direction,
                factor=edit_degree,
            )
        elif editing_direction in ganspace_directions:
            edit_direction = ganspace_directions[editing_direction]
            img_edit, edit_latents = editor.apply_ganspace(
                latent_codes[0].unsqueeze(0).cuda(), ganspace_pca, [edit_direction]
            )
        # apply targeted adversarial attack
        img_edit, edit_latents = launch_guided_attack(
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
        )
        # Align the distortion map
        img_edit = torch.nn.functional.interpolate(
            torch.clamp(img_edit, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )
        res_align = net.grid_align(torch.cat((res, img_edit), 1))

        # Fusion
        conditions = net.residue(res_align)
        result, _ = net.decoder(
            [edit_latents],
            conditions,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
    elif app_mode == "No Guidance Adversarial Editing":
        # calculate the distortion map
        imgs, _ = net.decoder(
            [latent_codes[0].unsqueeze(0).cuda()],
            None,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
        for step in range(inner_loop - 1):
            imgs = torch.nn.functional.interpolate(
                torch.clamp(imgs, -1.0, 1.0),
                size=(256, 256),
                mode="bilinear",
                # align_corners=False,
            )
            print(imgs.shape)
            latent_codes = get_latents(net, imgs)
            print(latent_codes.shape)
            imgs, _ = net.decoder(
                [latent_codes.cuda()],
                None,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
            )
        res = x - torch.nn.functional.interpolate(
            torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )

        img_edit, edit_latents = launch_no_guidance_attack(
            ckpt_input,
            latent_codes[0].unsqueeze(0).detach().cpu(),
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
        )

        # Align the distortion map
        img_edit = torch.nn.functional.interpolate(
            torch.clamp(img_edit, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )
        res_align = net.grid_align(torch.cat((res, img_edit), 1))

        # Fusion
        conditions = net.residue(res_align)
        result, _ = net.decoder(
            [edit_latents],
            conditions,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
    result = torch.nn.functional.interpolate(result, size=(256, 256), mode="bilinear")
    evaluator = IDLoss(id_eval_model)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tensor2im(result[0]).save(os.path.join(output_path, os.path.basename(image_path)))
    # print(result[0].shape)
    # print(input_image.shape)
    # print("Final Loss: ", evaluator(result, input_image.unsqueeze(0).cuda())[0])
    return display_alongside_source_image(tensor2im(result[0]), tensor2im(input_image))
