from criteria.id_loss import EnsembleIdLostMulti as IDLoss
from criteria.lpips.lpips import LPIPS


import torch


def residual_attack(
    net,
    x,
    imgs,
    new_latent_codes,
    res,
    id_loss_model,
    id_eval_model,
    lr=0.016,
    l2_lambda=0.001,
    id_threshold=0.4,
):
    # Forward pass
    id_loss = IDLoss(id_loss_model)
    evaluator = IDLoss(id_eval_model)
    lpips = LPIPS()
    mse_loss = torch.nn.MSELoss()
    # Initialize perturbed_res with zeros
    perturbed_res = res.clone().to(net.opts.device)
    perturbed_res.requires_grad = True
    optimizer = torch.optim.RMSprop([perturbed_res], lr=lr)
    img_edit = torch.nn.functional.interpolate(
        torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
    )

    # Loop for num_iterations
    id_dist_to_src = 0
    while id_dist_to_src < id_threshold:
        perturbed_res_align = net.grid_align(torch.cat((perturbed_res, img_edit), 1))
        # Encode the perturbed residual
        conditions = net.residue(perturbed_res_align)

        output, _ = net.decoder(
            [new_latent_codes],
            conditions,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
        # print(output.requires_grad)
        output = torch.nn.functional.interpolate(
            output, size=(256, 256), mode="bilinear"
        )
        # Compute loss
        loss = (
            lpips(output, x)
            - id_loss(output, x.cuda())[0]
            + l2_lambda * ((perturbed_res - res) ** 2).sum()
        )

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        id_dist_to_src = evaluator(output, x.cuda())[0]
        print(id_dist_to_src)
    return output, perturbed_res