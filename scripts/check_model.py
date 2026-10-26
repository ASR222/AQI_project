import tensorflow as tf

model = tf.keras.models.load_model("aqi_model.keras")

print("✅ Model loaded")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)