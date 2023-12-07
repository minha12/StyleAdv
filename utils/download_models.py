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


def download_model(model_name, destination):
    if not os.path.isfile(destination):
        print(f"Downloading {model_name}...")
        gdown.download(google_drive_paths[model_name], destination, quiet=False)
    else:
        print(f"{model_name} already exists. Skipping download.")


# Ensure the directory for pretrained models exists
if not os.path.isdir("./pretrained_models/"):
    os.makedirs("./pretrained_models/")

# Download each model if it doesn't already exist
download_model(
    "stylegan_encoder_decoder", "pretrained_models/stylegan_encoder_decoder.pt"
)
download_model(
    "CurricularFace_Backbone", "pretrained_models/CurricularFace_Backbone.pth"
)
download_model("facenet", "pretrained_models/facenet.pth")
download_model("ir152", "pretrained_models/ir152.pth")
download_model("irse50", "pretrained_models/irse50.pth")
download_model("mobile_face", "pretrained_models/mobile_face.pth")
download_model(
    "mmod_human_face_detector", "pretrained_models/mmod_human_face_detector.dat"
)
download_model(
    "shape_predictor_68_face_landmarks",
    "pretrained_models/shape_predictor_68_face_landmarks.dat",
)
