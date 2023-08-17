from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import socket

if socket.gethostname() == 'login-4': 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.save_pretrained("~/askem_shared/omoll/models/clip-vit-base-patch32")
    processor.save_pretrained("~/askem_shared/omoll/models/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save('000000039769.jpg')

model = CLIPModel.from_pretrained("~/askem_shared/omoll/models/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("~/askem_shared/omoll/models/clip-vit-base-patch32")
image = Image.open('000000039769.jpg')

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)



