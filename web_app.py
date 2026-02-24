import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Load the model
try:
    print("Loading model 'best_model_xception.keras'...")
    model = tf.keras.models.load_model('best_model_xception.keras')
    print("Model loaded successfully.")
    
    # Try to get input shape dynamically
    try:
        input_shape = model.input_shape[1:3] 
        if input_shape == (None, None):
            input_shape = (299, 299)
    except:
        input_shape = (299, 299)

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    input_shape = (299, 299)

def predict_image(img):
    if model is None:
        return {"Error": "Model not loaded"}
    
    if img is None:
        return {"Error": "No image provided"}
        
    try:
        # Resize image
        img = img.resize((input_shape[1], input_shape[0])) # PIL expects (width, height)
        img_array = np.array(img)
        
        # Ensure it has 3 channels (RGB)
        if img_array.ndim == 2: # Grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4: # RGBA
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for Xception (values between -1 and 1)
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Handle output shapes
        output_shape = model.output_shape
        
        if output_shape == (None, 1) or predictions.shape[-1] == 1:
            prob = float(predictions[0][0])
            # Common binary convention: >0.5 -> Dog, <0.5 -> Cat
            return {"Dog 🐶": prob, "Cat 🐱": 1.0 - prob}
            
        elif output_shape == (None, 2) or predictions.shape[-1] == 2:
            # Common categorical convention: index 0->Cat, index 1->Dog (usually, depending on data generator)
            return {"Cat 🐱": float(predictions[0][0]), "Dog 🐶": float(predictions[0][1])}
            
        else:
            return {"Unknown class 0": float(predictions[0][0])}
            
    except Exception as e:
        return {"Error": str(e)}

# Create Gradio interface using a clean, modern design
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo"
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🐱🐶 Cat vs Dog Classifier Web Service</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload an image, and the <b>Xception</b> deep learning model will determine whether it's a Cat or a Dog!</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload an Image")
            predict_btn = gr.Button("Predict!", variant="primary", size="lg")
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=2, label="Prediction Confidence")
            
    predict_btn.click(fn=predict_image, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    print(f"Starting Gradio web server with expected input shape {input_shape}...")
    demo.launch(share=False)
