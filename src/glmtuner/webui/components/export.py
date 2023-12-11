from typing import Dict
import gradio as gr
from gradio.components import Component

from glmtuner.webui.utils import export_model


def create_export_tab(top_elems: Dict[str, Component]) -> Dict[str, Component]:
    with gr.Row():
        save_dir = gr.Textbox()
        max_shard_size = gr.Slider(value=10, minimum=1, maximum=100)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        export_model,
        [
            top_elems["lang"],
            top_elems["model_name"],
            top_elems["checkpoints"],
            top_elems["finetuning_type"],
            max_shard_size,
            save_dir
        ],
        [info_box]
    )

    return dict(
        save_dir=save_dir,
        max_shard_size=max_shard_size,
        export_btn=export_btn,
        info_box=info_box
    )
