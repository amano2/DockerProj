"""Page segmentation modes: 
O Orientation and script detection (OSD) only
1 Automatic page segmentation with OSD. 
2 Automatic page segmentation, but no OSD, or OCR.
3 Fully automatic page segmentation, but no OSD. (Default)
4 Assume a single column of text of variable sizes.
5 Assume a single uniform block of vertically aligned text.
6 Assume a single uniform block of text
7 Treat the image as a single text line.
8 Treat the image as a single word.
9 Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.

OCR Engine Mode
0 Legacy  engine only.
1 Neural nets LSTM engine only.
2 Legacy + LSTM engine.
3 Default, based on what is available. 
"""

import pytesseract
from pytesseract import Output
import cv2
import re

# Configuration for OCR
myconfig = r"--psm 4 --oem 3"

# Load image and preprocess
img = cv2.imread("passport-td2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# OCR Processing
data = pytesseract.image_to_data(gray, config=myconfig, output_type=Output.DICT)

extracted_text = []

# Loop through each box
amount_boxes = len(data['text'])
for i in range(amount_boxes):
    if int(data['conf'][i]) > 60:  # Filter low-confidence results
        text = data['text'][i]
        
        # Use regex to identify passport number format
        if re.match(r"[A-Za-z0-9]{1,}", text):
            extracted_text.append(text)
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Draw rectangle and overlay text with background
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x, y - 30), (x + w, y), (0, 0, 0), -1)  # Background
            img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Display
print("Extracted Text:", " ".join(extracted_text))
cv2.imshow("Image", img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

 


