import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import coremltools as ct
import os

def validate_blip2_conversion():
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        torch_dtype=torch.float32,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
    )
    
    print("\nCreating example inputs...")
    # Create example inputs that match our use case
    example_image = torch.randn(1, 3, 364, 364)  # Match our image size
    example_text = "Describe the contents of this screenshot in detail."
    
    # Process inputs
    inputs = processor(
        images=example_image,
        text=example_text,
        return_tensors="pt",
        padding=True
    )
    
    print("\nTesting model tracing...")
    try:
        # Try to trace the model
        traced_model = torch.jit.trace(model, (
            inputs.pixel_values,
            inputs.input_ids,
            inputs.attention_mask
        ))
        print("✓ Model tracing successful")
    except Exception as e:
        print(f"✗ Model tracing failed: {str(e)}")
        return False
    
    print("\nTesting Core ML conversion...")
    try:
        # Convert to Core ML with validation
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="pixel_values", shape=inputs.pixel_values.shape),
                ct.TensorType(name="input_ids", shape=inputs.input_ids.shape),
                ct.TensorType(name="attention_mask", shape=inputs.attention_mask.shape)
            ],
            compute_units=ct.ComputeUnit.ALL,  # Try to use all available compute units
            minimum_deployment_target=ct.target.iOS16,  # Set minimum deployment target
            convert_to="mlprogram",  # Use the newer ML Program format
            compute_precision=ct.precision.FLOAT32,  # Start with full precision
            validate=True  # Enable validation
        )
        print("✓ Core ML conversion successful")
        
        # Save the converted model
        model_path = "models/blip2.mlmodel"
        os.makedirs("models", exist_ok=True)
        mlmodel.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Test the converted model
        print("\nTesting converted model...")
        mlmodel.predict({
            "pixel_values": inputs.pixel_values.numpy(),
            "input_ids": inputs.input_ids.numpy(),
            "attention_mask": inputs.attention_mask.numpy()
        })
        print("✓ Model prediction successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Core ML conversion failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting BLIP-2 to Core ML conversion validation...")
    success = validate_blip2_conversion()
    
    if success:
        print("\n✓ All validation steps passed! The model can be converted to Core ML.")
    else:
        print("\n✗ Validation failed. The model cannot be converted to Core ML.") 