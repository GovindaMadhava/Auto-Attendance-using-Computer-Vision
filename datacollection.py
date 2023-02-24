import cv2
import os


# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_name = input('\n Name : ')


dirName = 'Training/' + face_name
dirName1 = 'Test/' + face_name
dirName2 = 'Validation/' + face_name

try:
    # Create target Directory
    os.mkdir(dirName)
    os.mkdir(dirName1)
    os.mkdir(dirName2)
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while (True):

    ret, img = cam.read()

    faces = face_detector.detectMultiScale(img, scaleFactor=1.3,minNeighbors=5)

    for (x, y, w, h) in faces:


        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        print(count)

        if count<=320:
            cv2.imwrite(
                "Training/" + str(face_name) +"/"+ str(count) + ".jpg",
                img[y:y + h, x:x + w])
            count += 1

            cv2.imshow('image', img)
        elif count<=360:
            cv2.imwrite(
                "Validation/" + str(face_name) + "/" + str(count) + ".jpg",
                img[y:y + h, x:x + w])
            count += 1

            cv2.imshow('image', img)
        elif count<=400:
            cv2.imwrite(
                "Test/" + str(face_name) + "/" + str(count) + ".jpg",
                img[y:y + h, x:x + w])
            count += 1

            cv2.imshow('image', img)




    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    elif count > 400:
        break

# Do a bit of cleanup
print("\n Exiting Program")
cam.release()
cv2.destroyAllWindows()