import gdown
import os
import zipfile

google_drive_paths = {
    "stylegan_encoder_decoder": "https://drive.google.com/uc?id=1sdhKWEuezdjC3bFMO2gJkJZJp5gevDzq",
    "CurricularFace_Backbone": "https://drive.google.com/uc?id=1eYohkbi8WXEusDFKRs4ZEp3HOiHLKW_D",
    "facenet": "https://drive.google.com/uc?id=1hZ27WlRuCrl7kJo9doOxypaXJcGCQld2",
    "ir152": "https://drive.google.com/uc?id=1_Lb3ElnWu7SL-Fh_yknXSAIA7S2l4iYn",
    "irse50": "https://drive.google.com/uc?id=1bjHgpn6o99CSrXT4-DH0lO283sRJMPHN",
    "mobile_face": "https://drive.google.com/uc?id=1FnkXR0Wv8YqCzJ9FYpprv2Xi4m_wmNhK",
    "mmod_human_face_detector": "https://drive.google.com/uc?id=1Kt_xhMJRvn4G5j3NeTH5J2O6XcCU5m2j",
    "shape_predictor_68_face_landmarks": "https://drive.google.com/uc?id=1eGlrpiHTrocCsP1KZiFR3XXVesenJ1T1",
    "test_images": "https://drive.google.com/uc?id=1KXx4G-kcB9iQUNJoj8-xcVIngdT3kKJA",
}


def download(model_name, destination):
    if not os.path.exists(destination):
        print(f"Downloading {model_name}...")
        gdown.download(google_drive_paths[model_name], destination, quiet=False)
    else:
        print(f"The destination '{destination}' already exists. Skipping download.")


def unzip_and_remove(zip_file_path, extract_to_folder):
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_folder)
    print(f"Extracted all contents to {extract_to_folder}")

    # Remove the ZIP file
    os.remove(zip_file_path)
    print(f"Removed ZIP file {zip_file_path}")


# Ensure the directory for pretrained models exists
if not os.path.isdir("./pretrained_models/"):
    os.makedirs("./pretrained_models/")

# Download each model if it doesn't already exist
download("stylegan_encoder_decoder", "pretrained_models/stylegan_encoder_decoder.pt")
download("CurricularFace_Backbone", "pretrained_models/CurricularFace_Backbone.pth")
download("facenet", "pretrained_models/facenet.pth")
download("ir152", "pretrained_models/ir152.pth")
download("irse50", "pretrained_models/irse50.pth")
download("mobile_face", "pretrained_models/mobile_face.pth")
download("mmod_human_face_detector", "pretrained_models/mmod_human_face_detector.dat")
download(
    "shape_predictor_68_face_landmarks",
    "pretrained_models/shape_predictor_68_face_landmarks.dat",
)
download(
    "test_images",
    "test_images.zip",
)

zip_file = "test_images.zip"  # The path to your downloaded ZIP file
output_folder = "test_imgs"  # The directory where you want to extract the contents

unzip_and_remove(zip_file, ".")
