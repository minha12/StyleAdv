import argparse
import os
import sys

# add parent folder to default path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from attacks.styleadv import styleadv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the styleadv_run function with options"
    )

    # Define the command-line options
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results",
        help="Path to the directory of output image",
    )
    parser.add_argument(
        "--target_image_path", type=str, default="", help="Path to the target image"
    )
    parser.add_argument(
        "--app_mode",
        type=str,
        choices=[
            "Residual Attack",
            "Guided Adversarial Editing",
            "No Guidance Adversarial Editing",
        ],
        default="Guided Adversarial Editing",
    )
    parser.add_argument(
        "--id_loss_model", default=["irse50"], nargs="+", help="ID loss model"
    )
    parser.add_argument(
        "--id_eval_model", default=["cur_face"], nargs="+", help="ID evaluation model"
    )
    parser.add_argument(
        "--editing_direction",
        type=str,
        default="age",
        help="Editing direction for guided editing",
    )
    parser.add_argument(
        "--edit_degree", type=float, default=-1.5, help="Degree of editing"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument(
        "--id_threshold", type=float, default=0.43, help="Threshold for ID loss"
    )
    parser.add_argument("--l2_lambda", type=float, default=0.01, help="L2 lambda")
    parser.add_argument("--id_lambda", type=float, default=2, help="ID lambda")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of each batch")

    parser.add_argument(
        "--ckpt_input", type=str, default="", help="Path to the StyleGAN checkpoint"
    )
    parser.add_argument(
        "--stylegan_size", type=int, default=1024, help="Resolution of StyleGAN"
    )
    parser.add_argument(
        "--lr_rampup", type=float, default=0.05, help="Learning rate ramp-up"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="edit",
        choices=["edit", "other_modes"],
        help="Mode for StyleGAN",
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="Truncation value for StyleGAN"
    )
    parser.add_argument(
        "--seed_value",
        type=int,
        default=-1,
        help="Seed value for random number generation",
    )
    parser.add_argument(
        "--return_latent",
        action="store_true",
        help="Whether to return the latent vector",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Option to choose to run in demo mode or script/batch mode",
    )
    parser.add_argument(
        "--align_image",
        action="store_true",
        help="Choosing whether using align util to aligning the image",
    )
    args = parser.parse_args()

    return args


def main(args):
    args = parse_args()

    styleadv(
        input_path=args.input_path,
        output_path=args.output_path,
        target_image_path=args.target_image_path,
        app_mode=args.app_mode,
        editing_direction=args.editing_direction,
        edit_degree=args.edit_degree,
        lr=args.lr,
        id_threshold=args.id_threshold,
        l2_lambda=args.l2_lambda,
        id_lambda=args.id_lambda,
        id_loss_model=args.id_loss_model,
        id_eval_model=args.id_eval_model,
        batch_size=args.batch_size,
        ckpt_input=args.ckpt_input,
        stylegan_size=args.stylegan_size,
        lr_rampup=args.lr_rampup,
        mode=args.mode,
        truncation=args.truncation,
        seed_value=args.seed_value,
        return_latent=args.return_latent,
        demo=args.demo,
        align_image=args.align_image,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
