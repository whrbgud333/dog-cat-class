import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model('best_model_xception.keras')
print("Model loaded.")
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)
