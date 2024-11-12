# AUTOMATIC-ATTENDANCE-SYSTEM
Developed an ML model to detect faces from a real-time video stream, recognize individuals using a pre-trained model, and record attendance using Computer Vision.
The accuracy and loss for test data is: 93.87755393981934 and 0.49356546998023987

Packages required: opencv2 

Follow the steps below for execution:
1. In the "datacollection.py" file, mention name of person who's picture needs to be added to the attendance registry. 
By default, 400 pictures will be recorded and stored by the program. This can be altered accordingly. 

2. "dataPreprocessing.py" file is used to get the data in npy format.

3. "trainingCNN.py" file is used to train the ML model on the pictures captured using the camera. 
We have made use of Convolutional Neural Networks for this purpose.

4. Observe the graph and choose the model, taking into consideration bias variance trade off.

5. "test.py" file to observe the perfomance of model on previously-not-recorded / new data.

6. Finally run the "attendance.py" file.

7. Check the output of the model in "Attendance.csv" file.
