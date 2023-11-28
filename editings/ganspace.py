import torch


def edit(latents, pca, edit_directions):
    print(len(edit_directions))
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strength in edit_directions:
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = torch.zeros(latent.shape).to("cuda")
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta_padded)
    return torch.stack(edit_latents)


def edit_at_once(latents, pca, edit_directions):
    # print(len(edit_directions))
    edit_latents = []
    start, end, strengths = edit_directions
    for latent in latents:
        edited_latent = latent.clone()
        edited_layer = edit_layer(latent[0], pca, strengths)
        # padded = torch.zeros(latent.shape).to("cuda")
        edited_latent[start:end] = edited_layer.repeat(int(end - start), 1)
        edit_latents.append(edited_latent)
    return torch.stack(edit_latents)


# def edit_layer(latent, pca, strengths):
#     lat_comp = pca["comp"].to("cuda")
#     lat_std = pca["std"].to("cuda")
#     w_centered = latent - pca["mean"].to("cuda")

#     # Project the latent codes into the PCA space.
#     w_coords = torch.matmul(w_centered, lat_comp.transpose(0, 1)) / lat_std

#     # Adjust the coordinates of the projection along each principal component by the specified strength.
#     for pca_idx, strength in enumerate(strengths):
#         w_coords[0, pca_idx] += strength

#     # Transform the adjusted projection back into the original latent space.
#     edited_latent = torch.matmul(w_coords * lat_std, lat_comp) + pca["mean"].to("cuda")
#     return edited_latent


def edit_layer(latent, pca, strengths):
    lat_comp = pca["comp"].to("cuda").squeeze(1)  # This should have shape (80, 512)
    lat_std = pca["std"].to("cuda")
    w_centered = latent - pca["mean"].to("cuda")

    # Project the latent codes into the PCA space.
    w_coords = torch.matmul(w_centered, lat_comp.transpose(0, 1)) / lat_std

    # Adjust the coordinates of the projection along each principal component by the specified strength.
    for pca_idx, strength in enumerate(strengths):
        w_coords[0, pca_idx] += strength

    # Transform the adjusted projection back into the original latent space.
    edited_latent = torch.matmul(w_coords * lat_std, lat_comp) + pca["mean"].to("cuda")
    return edited_latent


# def edit(latents, pca, edit_directions):
#     edit_latents = []
#     for latent in latents:
#         edit_latent = (
#             latent.clone()
#         )  # clone the latent code so as not to modify the original one
#         for pca_idx, start, end, strength in edit_directions:
#             delta = get_delta(pca, latent, pca_idx, strength)
#             delta_padded = torch.zeros(latent.shape).to("cuda")
#             delta_padded[start:end] += delta.repeat(end - start, 1)
#             edit_latent += delta_padded  # apply the change to the edit_latent
#         edit_latents.append(
#             edit_latent
#         )  # add the edited latent code to the list after all directions applied
#     return torch.stack(edit_latents)


def get_delta(pca, latent, idx, strength):
    w_centered = latent - pca["mean"].to("cuda")
    print(w_centered.shape)
    lat_comp = pca["comp"].to("cuda")
    lat_std = pca["std"].to("cuda")
    w_coord = (
        torch.sum(w_centered[0].reshape(-1) * lat_comp[idx].reshape(-1)) / lat_std[idx]
    )
    print(w_coord)
    delta = (strength - w_coord) * lat_comp[idx] * lat_std[idx]
    return delta
