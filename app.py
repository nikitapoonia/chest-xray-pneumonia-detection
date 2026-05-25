import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Save model
tl_model.save('pneumonia_classifier.keras')
print("Model saved!")

# Load and predict
model = load_model('pneumonia_classifier.keras')


# Preprocess uploaded image
def predict_xray(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.
    if img_array.ndim == 2:  
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array)[0][0])

    return {
        "PNEUMONIA": round(prob, 3),
        "NORMAL":    round(1 - prob, 3)
    }

demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type='pil', label="Upload Chest X-Ray"),
    outputs=gr.Label(label="Prediction"),
    title=" Pneumonia Detector",
    description="Upload a chest X-ray image to classify as Normal or Pneumonia.",
    examples=[["test_normal.jpg"], ["test_pneumonia.jpg"]]
)

demo.launch(share=True)    