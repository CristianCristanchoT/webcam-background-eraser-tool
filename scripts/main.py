import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

height_image = 720
wide_imagen = 1280
fps = 60

threshold = 0.3

cap = cv2.VideoCapture(0)
cap.set(3,wide_imagen)
cap.set(4,height_image)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentator = SelfiSegmentation()
fpsReader = cvzone.FPS()

background_images_path = '../demo_backgrounds'
list_backimg = os.listdir(background_images_path)


background_images_list = []

for background in list_backimg:
    img = cv2.imread(f'{background_images_path}/{background}')
    img = cv2.resize(img, (wide_imagen, height_image), interpolation = cv2.INTER_AREA)
    background_images_list.append(img)

background_index = 0

while True:

    _, image = cap.read()

    image_out = segmentator.removeBG(image, background_images_list[background_index],threshold=threshold)

    
    imageStacked = cvzone.stackImages([image, image_out], 2 ,1)
    _, imageStacked = fpsReader.update(imageStacked, color=(0,0,255))

    org_confidence_text = (20, 100)

    confidence_text = f'Confidence: {round(threshold,2)}'

    cv2.putText(img=imageStacked, text=confidence_text, org=org_confidence_text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=(0, 0, 255),thickness=2)

    cv2.imshow("Image", imageStacked)

    k = cv2.waitKey(1)

    if k==27:    # Esc key to stop
        break

    if k == ord('a'):

        if background_index > 0:

            background_index -= 1
        else:
            background_index = len(background_images_list)-1

    if k == ord('d'):

        if background_index < len(background_images_list)-1:
            background_index += 1
        
        else:
            background_index = 0

    if k == ord('w'):

        if threshold < 0.99:
            threshold += 0.01

    if k == ord('s'):

        if threshold > 0.01:
            threshold -= 0.01
