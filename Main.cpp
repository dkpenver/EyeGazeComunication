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
int thresh = 25;
int max_thresh = 255;
RNG rng(12345);
Mat frame;
/** Function Headers */
void detectAndDisplay(Mat& frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_normal_name = "haarcascade_eye.xml";
String eyes_cascade_right_name = "haarcascade_righteye_2splits.xml";
String eyes_cascade_left_name = "haarcascade_lefteye_2splits.xml";
String smile_cascade_name = "haarcascade_smile.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade_normal;
CascadeClassifier eyes_cascade_right;
CascadeClassifier eyes_cascade_left;
CascadeClassifier smile_cascade;
string window_name1 = "Capture1 - Face detection";
string window_name2 = "Capture2 - Face detection";
int HH = 179, HL = 146, VL = 15, VH = 113, SL = 3, SH = 57;
/** @function main */
int main(int argc, const char** argv)
{
	face_cascade.load(face_cascade_name);
	eyes_cascade_normal.load(eyes_cascade_normal_name);
	eyes_cascade_left.load(eyes_cascade_left_name);
	eyes_cascade_right.load(eyes_cascade_right_name);
	smile_cascade.load(smile_cascade_name);
	//VideoCapture::VideoCapture(1);
	VideoCapture cap;

	cv::Mat frame, eye_tpl;
	cv::Rect eye_bb;

	cap.open(1);
	
	//namedWindow("window_name1", WINDOW_AUTOSIZE);
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade_normal.load(eyes_cascade_normal_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	//double contrast;
	while (true)
	{
		cap >> frame;
		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{
			detectAndDisplay(frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}

		int c = waitKey(200);
		if ((char)c == 'c') { break; }
	}

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat& frame)
{
	std::vector<Rect> faces;
	Mat frame_gray, cannyOut;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (faces.empty() == false) {

		for (size_t i = 0; i < faces.size(); i++)
		{
			Point centerFace(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(frame, centerFace, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;

			//-- In each face, detect eyes
			eyes_cascade_normal.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
				int p = j;
				Rect cropped((faces[i].x + eyes[p].x), (faces[i].y + eyes[p].y + eyes[p].height / 4), eyes[p].width, eyes[p].height*0.5);
				Mat frameCropped = frame(cropped);
				Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1;
			}
			eyes_cascade_left.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, eye_center, radius, Scalar(0, 255, 0), 4, 8, 0);
				int p = j;
				Rect cropped((faces[i].x + eyes[p].x), (faces[i].y + eyes[p].y + eyes[p].height / 4), eyes[p].width, eyes[p].height*0.5);
				Mat frameCropped = frame(cropped);
				Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1;
			}
			
			eyes_cascade_right.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, eye_center, radius, Scalar(0, 0, 255), 4, 8, 0);
				int p = j;
				Rect cropped((faces[i].x + eyes[p].x), (faces[i].y + eyes[p].y + eyes[p].height / 4), eyes[p].width, eyes[p].height*0.5);
				Mat frameCropped = frame(cropped);
				Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1;
			}
			smile_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, eye_center, radius, Scalar(255, 255,0 ), 4, 8, 0);
				int p = j;
				Rect cropped((faces[i].x + eyes[p].x), (faces[i].y + eyes[p].y + eyes[p].height / 4), eyes[p].width, eyes[p].height*0.5);
				Mat frameCropped = frame(cropped);
				Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1;
			}
			/*if (eyes.empty() == false) {
				if (eyes.size() >= 2) {
					int p = 0;
					for (int j = 1; j < eyes.size(); j++) {
						if ((faces[i].y + eyes[j].y) < (faces[i].y + eyes[j - 1].y)) {
							p = j;
						}
					}
					Point centerEye(faces[i].x + eyes[p].x + eyes[p].width*0.5, faces[i].y + eyes[p].y + eyes[p].height*0.5);
					int radius = cvRound((eyes[p].width + eyes[p].height)*0.25);
					circle(frame, centerEye,radius, Scalar(255, 0, 0), 4, 8, 0);
					Mat eyeROI1 = faceROI(eyes[p]);
					*/
					

				
			
		}
	}
			
		
	imshow(window_name1, frame);









/*
	namedWindow("control", WINDOW_AUTOSIZE);
	namedWindow("Origin", WINDOW_AUTOSIZE);
	createTrackbar("H Low", "control", &HL, 179);
	createTrackbar("H High", "control", &HH, 179);
	createTrackbar("S Low", "control", &SL, 255);
	createTrackbar("S High", "control", &SH, 255);
	createTrackbar("V Low", "control", &VL, 255);
	createTrackbar("VHigh", "control", &VH, 255);
	std::vector<Rect> faces;
	Mat frame_gray, cannyOut;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (faces.empty() == false) {

		for (size_t i = 0; i < faces.size(); i++)
		{
			Point centerFace(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(frame, centerFace, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;

			//-- In each face, detect eyes
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));


			if (eyes.empty() == false) {
				if (eyes.size() >= 2) {
					int p = 0;
					for (int j = 1; j < eyes.size(); j++) {
						if ((faces[i].y + eyes[j].y) < (faces[i].y + eyes[j - 1].y)) {
							p = j;
						}
					}
					Point centerEye(faces[i].x + eyes[p].x + eyes[p].width*0.5, faces[i].y + eyes[p].y + eyes[p].height*0.5);
					int radius = cvRound((eyes[p].width + eyes[p].height)*0.25);
					//circle(frame, centerEye,radius, Scalar(255, 0, 0), 4, 8, 0);
					Mat eyeROI1 = faceROI(eyes[p]);
					Rect cropped((faces[i].x + eyes[p].x), (faces[i].y + eyes[p].y + eyes[p].height / 4), eyes[p].width, eyes[p].height*0.5);
					Mat frameCropped = frame(cropped);
					Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1;



					cvtColor(frameCropped, HSVImage, CV_BGR2HSV);
					//cvtColor(frameCropped, frameCropped, CV_BGR2Lab);
					cvtColor(frameCropped, greyImage, CV_RGB2GRAY);
					inRange(HSVImage, Scalar(HL, SL, VL), Scalar(HH, SH, VH), eyeWhites);
					cvtColor(frameCropped, out, CV_BGR2GRAY);

					//Canny(out, cannyOut, 10, 75, 3);



					//Mat eyeROI2 = faceROI(eyes[1]);


					//Mat canny_output;
					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;

					/// Detect edges using canny
					equalizeHist(out, out1);
					Canny(out1, canny_output, 20, 80, 3);
					/// Find contours
					findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

					/// Draw contours
					Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
					for (int i = 0; i< contours.size(); i++)
					{
						Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
						drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
					}
					//Canny(eyeROI1, cannyOut, 90, 180);
					resize(drawing, drawing, Size(300, 200), 2, 2);
					//imshow(window_name2, eyeROI2);
					imshow("Origin", drawing);
					resize(eyeWhites, eyeWhites, Size(300, 200), 2, 2);
					resize(out, out, Size(300, 200), 2, 2);
					//resize(cannyOut, cannyOut, Size(300, 200), 2, 2);
					addWeighted(out, 1, eyeWhites, .5, 0, test);
					imshow(window_name1, test);
					//cout << eyes[j] << endl;

				}
			}
		}
	}*/
}