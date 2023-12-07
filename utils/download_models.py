import gdown
import os

google_drive_paths = {
    "stylegan_encoder_decoder": "https://drive.google.com/uc?id=1sdhKWEuezdjC3bFMO2gJkJZJp5gevDzq",
    "CurricularFace_Backbone": "https://drive.google.com/uc?id=1eYohkbi8WXEusDFKRs4ZEp3HOiHLKW_D",
    "facenet": "https://drive.google.com/uc?id=1hZ27WlRuCrl7kJo9doOxypaXJcGCQld2",
    "ir152": "https://drive.google.com/uc?id=1_Lb3ElnWu7SL-Fh_yknXSAIA7S2l4iYn",
    "irse50": "https://drive.google.com/uc?id=1bjHgpn6o99CSrXT4-DH0lO283sRJMPHN",
    "mobile_face": "https://drive.google.com/uc?id=1FnkXR0Wv8YqCzJ9FYpprv2Xi4m_wmNhK",
    "mmod_human_face_detector": "https://drive.google.com/uc?id=1Kt_xhMJRvn4G5j3NeTH5J2O6XcCU5m2j",
    "shape_predictor_68_face_landmarks": "https://drive.google.com/uc?id=1eGlrpiHTrocCsP1KZiFR3XXVesenJ1T1",
}


if not os.path.isdir("./pretrained_models/"):
    os.makedirs("./pretrained_models/")

# StyleGAN2 encode/decode
gdown.download(
    google_drive_paths["stylegan_encoder_decoder"],
    "pretrained_models/stylegan_encoder_decoder.pt",
    quiet=False,
)

# Download CurricularFace_Backbone.pth
gdown.download(
    google_drive_paths["CurricularFace_Backbone"],
    "pretrained_models/CurricularFace_Backbone.pth",
    quiet=False,
)

# Download facenet.pth
gdown.download(
    google_drive_paths["facenet"],
    "pretrained_models/facenet.pth",
    quiet=False,
)

# Download ir152.pth
gdown.download(
    google_drive_paths["ir152"],
    "pretrained_models/ir152.pth",
    quiet=False,
)

# Download irse50.pth
gdown.download(
    google_drive_paths["irse50"],
    "pretrained_models/irse50.pth",
    quiet=False,
)

# Download mobile_face.pth
gdown.download(
    google_drive_paths["mobile_face"],
    "pretrained_models/mobile_face.pth",
    quiet=False,
)

# Download mmod_human_face_detector.dat
gdown.download(
    google_drive_paths["mmod_human_face_detector"],
    "pretrained_models/mmod_human_face_detector.dat",
    quiet=False,
)

# Download shape_predictor_68_face_landmarks.dat
gdown.download(
    google_drive_paths["shape_predictor_68_face_landmarks"],
    "pretrained_models/shape_predictor_68_face_landmarks.dat",
    quiet=False,
)
