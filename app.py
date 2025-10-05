from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Initialize Flask
app = Flask(__name__)

# Load BLIP model + processor
processor = BlipProcessor.from_pretrained("./blip_model")
model = BlipForConditionalGeneration.from_pretrained("./blip_model")

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", caption="No file uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", caption="No file selected")

        # Open image
        image = Image.open(file.stream).convert("RGB")

        # Preprocess + generate caption
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    return render_template("index.html", caption=caption)

if __name__ == "__main__":
    app.run(debug=True)
