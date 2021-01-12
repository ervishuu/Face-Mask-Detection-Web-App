# import packages
from flask import Flask, render_template, request, send_from_directory
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model

#load model
model =load_model("Mask_detection_new_model.h5")
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    # model accept below hight and width of the image
    img_width, img_hight = 200, 200
    global COUNT

    # Load the Cascade face Classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # parameters for text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (1, 1)
    class_lable = ' '
    # fontScale
    fontScale = 1  # 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2  # 1

    # read image from webcam

    cimg = request.files['image']
    cimg.save('static/User Inputs/{}.jpg'.format(COUNT))
    color_img = cv2.imread('static/User Inputs/{}.jpg'.format(COUNT))


    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)

    # take face then predict class mask or not mask then draw recrangle and text then display image
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x - 10, y - 10)
        img_count += 1
        color_face = color_img[y:y + h, x:x + w]  # color face
        cv2.imwrite('faces/input/%dface.jpg' % (img_count), color_face)
        img = cv2.imread('faces/input/%dface.jpg' % (img_count))
        img = cv2.resize(img, (img_width, img_hight))

        img = img_to_array(img) / 255
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)
        # print(pred_prob[0][0].round(2))
        pred = np.argmax(pred_prob)
        #
        if pred == 0:
            print("User with mask - predic = ", pred_prob[0][0])
            # class_lable = "Mask"
            color = (255, 0, 0)
            cv2.imwrite('faces/with_mask/%dface.jpg' % (img_count), color_face)
            cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Using cv2.putText() method
            # cv2.putText(color_img, class_lable, org, font,
            #             fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite('faces/with_mask/%dmask.jpg' % (img_count), color_img)

        else:
            print('user not wearing mask - prob = ', pred_prob[0][1])
            # class_lable = "No Mask"
            # color = (0, 255, 0)
            cv2.imwrite('faces/without_mask/%dface.jpg' % (img_count), color_face)
            cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Using cv2.putText() method
            # cv2.putText(color_img, class_lable, org, font,
            #             fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite('faces/with_mask/%dno_mask.jpg' % (img_count), color_img)


    x = round(pred_prob[0][0], 2)
    y = round(pred_prob[0][1], 2)
    preds = np.array([x,y])
    cv2.imwrite('output images/{}.jpg' .format(COUNT), color_img)

    COUNT += 1

    return render_template('prediction.html', data=preds)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('output images', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



