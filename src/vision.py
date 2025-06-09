import os
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dotenv import load_dotenv

class VisionAnalyzer:
    def __init__(self):
        self.device = "cpu"  # Force CPU usage for better compatibility
        print(f"Using device: {self.device}")
        
        try:
            print("Loading processor...")
            self.processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b-coco",
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
            )
            
            print("Loading model...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b-coco",
                torch_dtype=torch.float32,  # Needed because MPS doesn't support float16 well
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
            ).to(self.device)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def analyze_image(self, image_path):
        """Analyze an image using BLIP-2 model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Description of the image content
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text="Describe the contents of this screenshot in detail. What text is visible? What commands or output are shown? What is the layout and appearance?",
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            print("Inputs prepared, shape:", {k: v.shape for k, v in inputs.items()})
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=300,  # Increased for longer descriptions
                    num_beams=7,
                    min_length=100,  # Increased minimum length
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=0.7,
                    do_sample=False,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3  # Prevent repetitive phrases
                )
            
            print("Generated IDs shape:", generated_ids.shape)
            result = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            print("Raw result:", result)
            
            if not result.strip():
                return "The model generated an empty response. Please try again."
            
            return result
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise 