import sys

import torchvision.transforms as transforms
import gradio as gr

from attacks.styleadv import (
    ganspace_directions,
    interfacegan_directions,
    styleadv,
)


from utils import ensure_checkpoint_exists
from utils.align import align_face


from utils.common import get_file_list
from utils.common import get_path_list

print("Model successfully loaded!")
# Gradio app
css = """ 
#main_col {
overflow-y: scroll;
}
"""
# max-height: 800px;
with gr.Blocks(css=css) as app:
    gr.Markdown("# Adversarial Editing App")
    result_image_output = gr.Image(type="pil", label="Result Image")
    generate_button = gr.Button("Generate")
    with gr.Row(elem_id="main_col"):
        with gr.Column():
            image_default_dir = "test_imgs/"
            image_files = get_path_list(image_default_dir)
            image_path_input = gr.Dropdown(
                image_files,
                label="Input Path",
                value="test_imgs/00635.jpg",
            )
            image_output_path = gr.Textbox(label="Output Path", value="./results")
            mode_dropdown = gr.Dropdown(
                choices=[
                    "Guided Adversarial Editing",
                    "Residual Attack",
                ],
                value="Guided Adversarial Editing",
                label="Mode",
            )
            # use_conditions = gr.Radio(
            #     [True, False],
            #     label="Use fine-gained details",
            #     value=True,
            # )
            # outer_loop = gr.Number(
            #     value=1,
            #     label="Outer encoding loop",
            #     precision=0,
            # )
            # inner_loop = gr.Number(
            #     value=1,
            #     label="Inner encoding loop",
            #     precision=0,
            # )
            with gr.Accordion("Options for aversarial editing", open=True):
                image_files = get_path_list("test_imgs/")
                image_path_tarid = gr.Dropdown(
                    image_files,
                    label="Target ID",
                    value="",
                )
                editing_direction_dropdown = gr.Dropdown(
                    choices=list(interfacegan_directions.keys())
                    + list(ganspace_directions.keys()),
                    label="Editing Direction",
                    value="age",
                )
                edit_degree_slider = gr.Slider(
                    minimum=-3, maximum=3, step=0.1, value=-1.5, label="Edit Degree"
                )
            # with gr.Accordion("Options for aversarial editing", open=False):

        with gr.Column():
            # ckpt_input = os.path.join(ckpt_default_dir, ckpt_input)
            id_threshold = gr.Number(0.43, label="Threshold for ID Loss")
            id_lambda_input = gr.Number(2, label="ID Lambda")
            l2_lambda_input = gr.Number(0.01, label="L2 Lambda")
            lr_input = gr.Number(0.05, label="Learning Rate")
            # n_batch = gr.Number(1, label="Number of batch", precision=0)
            # batch_size = gr.Number(1, label="Batch size", precision=0)
            id_loss_model = gr.CheckboxGroup(
                choices=["ir152", "irse50", "mobile_face", "facenet", "cur_face"],
                label="Attacker Model",
                value=["irse50"],
            )
            id_eval_model = gr.CheckboxGroup(
                choices=["ir152", "irse50", "mobile_face", "facenet", "cur_face"],
                label="Victim Model",
                value=["cur_face"],
            )

    generate_button.click(
        styleadv,
        inputs=[
            image_path_input,
            image_output_path,
            image_path_tarid,
            mode_dropdown,
            editing_direction_dropdown,
            edit_degree_slider,
            lr_input,
            id_threshold,
            l2_lambda_input,
            id_lambda_input,
            id_loss_model,
            id_eval_model,
            # n_batch,
            # batch_size,
        ],
        outputs=result_image_output,
    )

app.launch()
