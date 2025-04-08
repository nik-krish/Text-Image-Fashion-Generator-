from one import generator, clip_processor, clip_model, generate_from_prompt
from torchvision.transforms import ToPILImage

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Allow requests from React

@app.route("/", methods=["GET"])
def home():
    return "Backend is running. Use POST /generate to generate an image."


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    generated_img = generate_from_prompt(generator, prompt)
    img = ToPILImage()(generated_img.squeeze(0).cpu())

    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(port=5000)
