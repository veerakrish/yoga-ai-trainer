import tensorflow as tf
import tf2onnx
import onnx

def convert_model(model_path, output_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    
    # Save the ONNX model
    onnx.save(onnx_model, output_path)

# Convert all models
models = ['hatha_model.h5', 'surya_model.h5', 'vinyasana_model.h5']
for model_name in models:
    input_path = model_name
    output_path = model_name.replace('.h5', '.onnx')
    convert_model(input_path, output_path)
    print(f"Converted {model_name} to {output_path}")
