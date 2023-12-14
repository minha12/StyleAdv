import numpy as np
import torch
from torch import nn

from models.facial_recognition.model_irse import IR_101
import torch.nn.functional as F

import models.facial_recognition.irse as irse
import models.facial_recognition.facenet as facenet
import models.facial_recognition.ir152 as ir152


class IdLostMulti(nn.Module):
    def __init__(self, model_name):
        super(IdLostMulti, self).__init__()
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fr_model = self.load_fr_model()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool_112 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.face_pool_160 = torch.nn.AdaptiveAvgPool2d((160, 160))

    def load_fr_model(self):
        fr_model = None

        if self.model_name == "ir152":
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load("./pretrained_models/ir152.pth"))
            # print("Loaded IR152 model")
        elif self.model_name == "irse50":
            fr_model = irse.Backbone(50, 0.6, "ir_se")
            fr_model.load_state_dict(torch.load("./pretrained_models/irse50.pth"))
            # print("Loaded IRSE50 model")
        elif self.model_name == "mobile_face":
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load("./pretrained_models/mobile_face.pth"))
            # print("Loaded MobileFace model")
        elif self.model_name == "facenet":
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
            fr_model.load_state_dict(torch.load("./pretrained_models/facenet.pth"))
            # print("Loaded Facenet model")
        elif self.model_name == "cur_face":
            fr_model = IR_101(input_size=112)
            fr_model.load_state_dict(
                torch.load("./pretrained_models/CurricularFace_Backbone.pth")
            )
            # print("Loaded CurricularFace model")

        fr_model.to(self.device)
        fr_model.eval()

        return fr_model

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        if self.model_name == "facenet":
            x = self.face_pool_160(x)  # convert to 160 x 160
        else:
            x = self.face_pool_112(x)
        x_feats = self.fr_model(x)
        if self.model_name == "ir152":
            x_feats = F.normalize(x_feats, p=None, dim=1)

        return x_feats  # torch.Size([x.shape[0], 512])

    def forward(self, y_hat, y):  # y_hat have the gradient
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count


class EnsembleIdLostMulti(nn.Module):
    def __init__(self, model_names):
        super(EnsembleIdLostMulti, self).__init__()
        self.models = nn.ModuleList(
            [IdLostMulti(model_name) for model_name in model_names]
        )

    def forward(self, y_hat, y):
        losses = []
        sim_improvements = []
        for model in self.models:
            loss, sim_improvement = model(y_hat, y)
            losses.append(loss)
            sim_improvements.append(sim_improvement)
        # Take the average of the features
        avg_loss = torch.mean(torch.stack(losses))
        sim_improvement = np.mean(sim_improvements)
        return avg_loss, sim_improvement
