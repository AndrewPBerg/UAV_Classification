import visualkeras
from tensorflow.keras.applications import EfficientNetB0
from PIL import ImageFont
import tensorflow as tf

# Workaround for the shape mismatch bug in EfficientNetB0
# Create the model without weights first, then try to load compatible weights
try:
    # First attempt: Try creating without weights to avoid the shape mismatch
    model = EfficientNetB0(weights=None, input_shape=(224, 224, 3))
    print("Created EfficientNetB0 model without pre-trained weights")
    
    # Try to load weights manually if the architecture creation succeeds
    try:
        # This will attempt to download and load ImageNet weights
        imagenet_model = EfficientNetB0(weights='imagenet', input_shape=(224, 224, 3))
        # If successful, use the ImageNet model
        model = imagenet_model
        print("Successfully loaded ImageNet weights")
    except Exception as weight_error:
        print(f"Could not load ImageNet weights: {weight_error}")
        print("Using model without pre-trained weights for visualization")
        
except Exception as e:
    print(f"Failed to create EfficientNetB0: {e}")
    # Fallback: Use a simpler model for visualization
    from tensorflow.keras.applications import MobileNetV2
    print("Falling back to MobileNetV2 for visualization")
    model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Create a font object.
# The default font might be too small on some systems, so you can specify a path to a TrueType font file
# for better control over the text size. For example:
# font = ImageFont.truetype("arial.ttf", 24)
font = ImageFont.load_default()

# Generate the visualization of the model's architecture.
# `layered_view` creates a layered, top-down view of the model.
# `legend=True` adds a legend to the visualization.
# `to_file` specifies the output file path.

# Determine the output filename based on the model type
model_name = model.name if hasattr(model, 'name') else type(model).__name__
output_file = f'/Users/applefella/Documents/UAV_Classification/UAV_Classification/src/images/{model_name}_visualization.png'


visualkeras.layered_view(
    model,
    legend=True,
    font=font,
    to_file=output_file,
    scale_xy=2,            # Make the blocks larger for better visibility
    scale_z=1,             # Depth scaling
    draw_volume=True,      # Show 3D volume for each layer
    spacing=40,            # Increase spacing between layers
    max_z=16,              # Limit the max depth of blocks
    # alpha=0.8,             # Set transparency for better layer distinction
    background_fill=(255,255,255,255),  # White background
    # draw_funnel and type_ignore are not valid arguments for visualkeras.layered_view as of the latest official release.
    # See: https://github.com/paulgavrikov/visualkeras/blob/master/visualkeras/layered.py
    # Only use supported flags to avoid hallucinations.
)

print(f"Model visualization saved to '{output_file}'")

