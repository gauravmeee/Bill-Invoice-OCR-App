# Upload invoices Image , and Show text bounded Image and Extract Data Inside it. 
# ('/') Get ->  index.html -> ('/') Post -> output.html -> ('/output.jpg')

from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for uploading the image
@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['image']
    temp_filename = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_filename)
    # Perform OCR processing using Tesseract and OpenCV
    image = cv2.imread(temp_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    # Save the output image with bounding boxes
    output_image = draw_bounding_boxes(image, text)
    output_filename = os.path.join(os.getcwd(), 'output.jpg')
    cv2.imwrite(output_filename, output_image)
    # Redirect to the output page
    return render_template('output.html', text=text)

# Route for serving the output image
@app.route('/output.jpg', methods=['GET'])
def serve_output_image():
    output_filename = os.path.join(os.getcwd(), 'output.jpg')
    return send_file(output_filename, mimetype='image/jpeg')


# def draw_bounding_boxes(image, text):
#     d = pytesseract
#     # Implement your bounding box drawing logic here
#     # For demonstration purposes, I'll just return the original image
#     return image
#--------------------sourabh function ------------------------------------
def draw_bounding_boxes(img, text):
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)