#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
Mat cameraFeed;
Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name1 = "Capture1 - Face detection";
string window_name2 = "Capture2 - Face detection";
int HH = 179, HL = 0, VL = 0, VH = 255, SL = 0, SH = 255;
/** @function main */
int main(int argc, const char** argv)
{
	VideoCapture capture;
	capture.open(1);
	Mat frame;
	
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	
		while (true)
		{
			capture.read(frame);
			//frame = cameraFeed;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(1);
			if ((char)c == 'c') { break; }
		}
	
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{

	namedWindow("control",WINDOW_AUTOSIZE);
	namedWindow("Origin", WINDOW_AUTOSIZE);
	createTrackbar("H Low", "control", &HL, 179);
	createTrackbar("H High", "control", &HH, 179);
	createTrackbar("S Low", "control", &SL, 255);
	createTrackbar("S High", "control", &SH, 255);
	createTrackbar("V Low", "control", &VL, 255);
	createTrackbar("VHigh", "control", &VH, 255);
	std::vector<Rect> faces;
	Mat frame_gray,cannyOut;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(80, 80));
	if (faces.empty() == false) {

		for (size_t i = 0; i < faces.size(); i++)
		{
			Point centerFace(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;

			//-- In each face, detect eyes
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));


			if (eyes.empty() == false) {
				Point centerEye(faces[i].x + eyes[0].x + eyes[0].width*0.5, faces[i].y + eyes[0].y + eyes[0].height*0.5);
				int radius = cvRound((eyes[0].width + eyes[0].height)*0.25);
				//circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
				Mat eyeROI1 = faceROI(eyes[0]);
				Rect cropped((faces[i].x + eyes[0].x), faces[i].y + eyes[0].y, eyes[0].width, eyes[0].height);
				Mat frameCropped = frame(cropped);
				Mat HSVImage, eyeWhites,test ,out;

				
				
				cvtColor(frameCropped, HSVImage,CV_BGR2HSV);
				//cvtColor(frameCropped, frameCropped, CV_BGR2Lab);

					inRange(HSVImage, Scalar(HL, SL, VL), Scalar(HH, SH, VH), eyeWhites);
					cvtColor(frameCropped, out, CV_BGR2GRAY);
					resize(eyeWhites, eyeWhites, Size(300, 300), 2, 2);
					resize(out,out, Size(300, 300), 2, 2);
					addWeighted(out, 1, eyeWhites, .5, 0, test);
					imshow(window_name1, test);
					imshow("Origin", eyeWhites);
				//Mat eyeROI2 = faceROI(eyes[1]);
				

				/*Mat canny_output;
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;

				/// Detect edges using canny
				Canny(eyeROI1, canny_output, thresh, thresh * 2, 3);
				/// Find contours
				findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

				/// Draw contours
				Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
				for (int i = 0; i< contours.size(); i++)
				{
					Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
					drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
				}
				//Canny(eyeROI1, cannyOut, 90, 180);*/
				//
				
				//imshow(window_name2, eyeROI2);
				//cout << eyes[j] << endl;
			}
		}
	}	
}