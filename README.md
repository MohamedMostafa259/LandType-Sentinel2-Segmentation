# LandType-Sentinel2-Segmentation

Semantic segmentation of satellite imagery from Sentinel-2 to classify land types such as water, vegetation, roads, buildings, and unpaved land. I performed pixel-level semantic segmentation to accurately label different land cover types.

## Dataset Overview

The primary dataset contains **72 tiles** of **Dubai**, each segmented into **6 semantic classes** (water, vegetation, roads, buildings, unpaved land, and unlabeled). These were obtained from MBRSC satellites and manually labeled.

**Source**: [Kaggle – Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

## Approach

### Preprocessing

* **Patchification**: Cropped the original 72 large tiles into 1,305 patches of size 256×256 pixels.
* **Normalization**: Applied max normalization (rescaling by 1/255.0) to bring pixel values into the \[0.0, 1.0] range.

### Model Development

* **Custom U-Net**: Implemented U-Net from scratch based on [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).
* **Pretrained U-Net**: Used `segmentation_models.Unet(...)` with:

  * **Encoder**: `EfficientNetB4`
  * **Weights**: `ImageNet`

### Evaluation & Visualization

* Compared predictions against ground truth and input images.

    ![original_groundTruth_prediction_comparison](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/original_groundTruth_prediction_comparison.png?raw=true)

<br>

* Performed diagnostics (Loss, IoU score, Accuracy) over epochs during training to assess segmentation quality.

    ![model_diagnostics_over_epochs](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/model_diagnostics_over_epochs.png?raw=true)

    Validation Accuracy: **89.5%**

    Validation IoU Score: **0.68**

<br>

### Generalization Testing

* Tested on unseen tiles from the **Inria Aerial Image Labeling dataset**.

    ![pred2_InriaAerialImageLabeling](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/test/pred2_InriaAerialImageLabeling.png?raw=true)


* Also evaluated generalization on samples taken from **Google Maps**.

    ![pred1_GoogleMaps](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/test/pred1_GoogleMaps.png?raw=true)


### [Deployment](https://huggingface.co/spaces/mohamedmostafa259/Satellite-Segmentation-Prediction)

- **Model Inference:**

    The `app.py` hosts a Gradio interface allowing you to upload Sentinel-2 images and get real-time land type segmentation masks with a clear legend explaining color codes:

    ![legend](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/legend.png?raw=true)

<br>

- Built a **Gradio-based** interactive web app for real-time inference.

    ⚠️ **Note:** Hugging Face Spaces may sleep after periods of inactivity. If the link doesn't load, feel free to open an issue or contact me to restart the service. But for now, you can look at the images below :)

    ![app_screenshot1](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/app_screenshot1.png?raw=true)

    ![app_screenshot2](https://github.com/MohamedMostafa259/LandTypeClassification-Sentinel2/blob/main/visulas/app_screenshot2.png?raw=true)


## Future Improvements

* Integrate more spectral bands from Sentinel-2 (beyond RGB) for better accuracy
* Experiment with advanced segmentation heads like DeepLabV3+ or attention modules
* Build a more sophisticated web deployment with scalable API backends
* Build more sophisticated preprocessing pipelines, including heavy data augmentation to see its effect.

## Repository Structure

```
├── data/       →   Untracked
├── models/     →   Untracked
├── notebooks/
│   ├── dataPreparation.ipynb                           # Exploratory Data Analysis & preprocessing
│   ├── modelTraining.ipynb                             # Model architecture, training, evaluation
│   └── trackingWeightsAndBiases.ipynb                  # Experiment tracking with Weights & Biases
├── visulas/
│   ├── app_screenshot1.png                             # screenshot of the deployed Gradio app on Hugging Face
│   ├── app_screenshot2.png                  
│   ├── legend.png                                      # table containing color legends for each class
│   ├── original_groundTruth_prediction_comparison.png  # Side-by-side image vs prediction
│   ├── model_diagnostics_over_epochs.png               # Training curves (loss, accuracy, IoU)
│   └── test/...                                        # External satellite images used for testing generalization
│   
├── utils.py                                            # Utility functions (diagnostics plotting)
├── app.py                                              # Gradio app
├── requirements.txt                                    # Utility functions (diagnostics plotting)
├── .gitignore                                      
└── README.md                                           # This file
````

<br>

**Feel free to open issues or contribute!**