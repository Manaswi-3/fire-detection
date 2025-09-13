from flask import Flask, render_template, request, redirect, url_for
import os
from scripts.detect import detect_fire

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Detect fire in the image
        result = detect_fire(file_path)
        return render_template("index.html", result=result, image_path=file_path)

    return render_template("index.html", result=None, image_path=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
