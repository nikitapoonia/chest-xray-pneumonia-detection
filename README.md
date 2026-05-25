 #  Chest X-Ray Pneumonia Detection

A deep learning project that detects **Pneumonia from Chest X-Ray images** using CNN and Transfer Learning (VGG16).



##  Dataset
- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,856 images
- **Classes:** Normal vs Pneumonia
- **Split:** Train / Validation / Test



##  Results

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| AUC | ~0.93 |
| Model | VGG16 Transfer Learning |



##  Project Structure
chest-xray-pneumonia/
├── train_model.ipynb   
├── app.py              
├── requirements.txt    
└── README.md



##  How It Works

1. **Data Augmentation** — rotation, zoom, flip to increase variety
2. **Transfer Learning** — VGG16 pre-trained on ImageNet
3. **Class Weights** — handles imbalanced dataset (3x more Pneumonia images)
4. **Grad-CAM** — heatmap showing where model looks in the X-ray
5. **Gradio App** — upload any X-ray and get instant prediction



##  Key Concepts Used

- Convolutional Neural Networks (CNN)
- Transfer Learning (VGG16)
- Data Augmentation
- Class Imbalance Handling
- Grad-CAM Explainability
- EarlyStopping & ReduceLROnPlateau Callbacks



##  Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```



##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Gradio](https://img.shields.io/badge/Gradio-latest-purple)



##  Sample Grad-CAM Output

> Red areas show where the model focuses to make its prediction.
> For Pneumonia images, heatmap highlights infected lung regions.



##  Author
**Nikita Poonia**
- GitHub: https://github.com/nikitapoonia/chest-xray-pneumonia-detection

