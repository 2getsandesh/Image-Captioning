import requests
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
 
# conditional image captioning
text = "a photography of"
def caption_generation_conditional(image,text):
    
    inputs = processor(images=image, text=text, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# unconditional image captioning
def caption_generation_unconditional(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def caption_image(image):
    try:
        caption1 = caption_generation_conditional(image)
        caption2 = caption_generation_unconditional(image)
        return caption1, caption2
    except Exception as e:
        return f"An error occurred: {str(e)}"

iface = gr.Interface(
    fn = caption_image,
    inputs=gr.Image(type="pil"),
    outputs=["text","text"],
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption.",
    
).launch()


