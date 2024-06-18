import requests
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load a sample image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
def caption_generation_conditional(image, text):
    inputs = processor(images=image, text=text, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Unconditional image captioning
def caption_generation_unconditional(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Caption image function
def caption_image(image):
    try:
        # Provide a default text prompt for conditional caption generation
        text_prompt = "a photography of"
        caption1 = caption_generation_conditional(image, text_prompt)
        caption2 = caption_generation_unconditional(image)
        return caption1, caption2
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

# Create Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Conditional Caption"), gr.Textbox(label="Unconditional Caption")],
    title="Image Captioning with BLIP",
    description="Upload an image to generate both conditional and unconditional captions."
)

iface.launch()



