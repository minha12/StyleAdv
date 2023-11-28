from models.StyleGANWrapper import StyleGANWrapper


import torch


from argparse import Namespace


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