import gradio as gr
from engine import Engine
from model_training import *

def doodle_search(img, model_choice, topk):
    pretrained_paths = {
        "V1 MLP": ("", ""),
        "V2 CNN": ("", ""),
        "V3 CNN + Contrastive": ("", ""),
        "V4 CNN + Multi-contrastive": ("", ""),
        "V4 ConvNeXT + Multi-contrastive": ("", ""),
    }

    models = {
        "V1 MLP": (None, None),
        "V2 CNN": (None, None),
        "V3 CNN + Contrastive": (None, None),
        "V4 CNN + Multi-contrastive": (None, None),
        "V4 ConvNeXT + Multi-contrastive": (None, None),
    }

    doodle_path, real_path = pretrained_paths[model_choice]
    doodle_model_class, real_model_class = models[model_choice]
    doodle_model = load_model(doodle_model_class, doodle_path)
    real_model = load_model(real_model_class, real_path)

    real_val_set = RealDataset(train=False)
    engine = Engine(real_val_set, doodle_model, real_model)
    results = engine.query(img, topk=topk)

    return results

demo = gr.Interface(
    fn=sketch_recognition,
    inputs=[
        gr.inputs.Image(shape=(200, 200), image_mode="L", invert_colors=False, source="canvas"), 
        gr.inputs.Dropdown(
            ["V1 MLP", "V2 CNN", "V3 CNN + Contrastive", "V4 CNN + Multi-contrastive", "V5 ConvNeXT + Multi-contrastive"], 
            label="Choose your model"
        ),
        gr.inputs.Slider(minimum=5, maximum=20, step=1, label="Top K Best Matches")
    ],  
    outputs="label")

if __name__ == "__main__":
    demo.launch()