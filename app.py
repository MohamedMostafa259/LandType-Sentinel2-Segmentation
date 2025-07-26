import gradio as gr
from PIL import Image
import numpy as np
import segmentation_models as sm
from matplotlib.colors import ListedColormap

import tensorflow as tf
from tensorflow.keras import backend as K

if not hasattr(K, 'sigmoid'):
	K.sigmoid = tf.nn.sigmoid

sm.set_framework('tf.keras')
print("Framework set to:", sm.framework())

tf.random.set_seed(42)

IMAGE_PATCH_SIZE = 256

color_list = [
    '#E2A929',  # Class 0 → Water
    '#8429F6',  # Class 1 → Land (unpaved area)
    '#6EC1E4',  # Class 2 → Road
    '#3C1098',  # Class 3 → Building
    '#FEDD3A',  # Class 4 → Vegetation
    '#9B9B9B'   # Class 5 → Unlabeled
]
class_labels = [
    "Water", "Land (unpaved area)", "Road", "Building", "Vegetation", "Unlabeled"
]
cmap = ListedColormap(color_list)

satellite_model = tf.keras.models.load_model(
	'models\satellite_segmentation_model_pretraining.keras',
	compile=False)

def process_input_image(input_image):
	image = input_image.resize((256, 256))
	image = np.array(image)
	image = np.expand_dims(image, axis=0)  # make it a batch of size 1
	image = image / 255.0

	predicted_image = satellite_model.predict(image)
	# take argmax and then remove the batch dimension
	predicted_image = np.argmax(predicted_image, axis=-1)[0, :]
	predicted_image = predicted_image.astype(np.uint8)

	# ListedColormap(predicted_image) returns a float array in RGBA format: Range is [0.0, 1.0]
	rgb_mask = (cmap(predicted_image)[:, :, :3] * 255).astype(np.uint8)

	return Image.fromarray(rgb_mask)

legend_html = """
<div style='display: flex; flex-direction: column; gap: 6px;'>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #E2A929; border: 1px solid #000;'></div> Water</div>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #8429F6; border: 1px solid #000;'></div> Land (unpaved area)</div>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #6EC1E4; border: 1px solid #000;'></div> Road</div>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #3C1098; border: 1px solid #000;'></div> Building</div>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #FEDD3A; border: 1px solid #000;'></div> Vegetation</div>
  <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 18px; height: 18px; background-color: #9B9B9B; border: 1px solid #000;'></div> Unlabeled</div>
</div>
"""

with gr.Blocks(title="Satellite Image Segmentation") as interface:
    gr.Markdown("## Satellite Segmentation Predictor")
    gr.HTML(legend_html)

    with gr.Row():
        input_img = gr.Image(type="pil", label=f"Input Image ({IMAGE_PATCH_SIZE}x{IMAGE_PATCH_SIZE})", image_mode="RGB")
        output_img = gr.Image(type="pil", label="Predicted Masked Image")

    submit_button = gr.Button("Predict")
    submit_button.click(fn=process_input_image, inputs=input_img, outputs=output_img)

interface.launch()