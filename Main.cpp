
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <array>
#include <ctime>
#include <string>

#define MAX_DATE 12

using namespace std;
using namespace cv;

Mat cameraFeed;
Mat src; Mat src_gray;
int thresh = 25;
int max_thresh = 255;
RNG rng(12345);
Mat frame;

/** Function Headers */
Mat detectFace(Mat& frame);
Mat eyeLocation(Mat& frame, Mat& frame_grey, std::vector<Rect> faces, size_t location);
string get_time(void);
void training_image_left(Mat eye);
void training_image_right(Mat eye); 
void training_image_up(Mat eye);
void training_image_down(Mat eye);
void direction(Mat& frameCropped);
Mat capture_image();
void comparison();

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
std::string window_name1 = "Capture1 - Face detection";
std::string window_name2 = "Capture2 - Face detection";
std::string window_name3 = "Capture3 - Face detection";

int HH = 179, HL = 146, VL = 15, VH = 113, SL = 3, SH = 57;
/** @function main */
int main()
{

	face_cascade.load(face_cascade_name);
	eyes_cascade_normal.load(eyes_cascade_normal_name);
	eyes_cascade_left.load(eyes_cascade_left_name);
	eyes_cascade_right.load(eyes_cascade_right_name);
	smile_cascade.load(smile_cascade_name);
	//VideoCapture::VideoCapture(1);

	namedWindow("Origin", WINDOW_AUTOSIZE);
	namedWindow("window_name1", WINDOW_AUTOSIZE);
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade_normal.load(eyes_cascade_normal_name)) { printf("--(!)Error loading\n"); return -1; };

	Mat left, right, up, down, loop;
	waitKey(0);
	left = capture_image();
	training_image_left(left);
	imshow("Origin", left);
	waitKey(0);
	right = capture_image();
	training_image_left(right);
	imshow("Origin", right);
	waitKey(0);
	up = capture_image();
	training_image_left(up);
	imshow("Origin", up);
	waitKey(0);
	down = capture_image();
	training_image_left(down);
	imshow("Origin", down);
	waitKey(0);
	cout << "loop begins" << endl << endl;
	/*while (true){
		loop=capture_image(); 
		imshow("Origin", loop);
		waitKey(1);
	}*/
}

Mat capture_image() {
	cout << "capture image" << endl;
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	Mat frame,result, input;
	cap.read(frame);
	resize(frame, input, Size(300, 200), 2, 2);
	imshow("window_name1", input);
	waitKey(1);
	cout << "image taken" << endl;
	//-- 3. Apply the classifier to the frame
	if (!frame.empty())
	{
		result= detectFace(frame);
	}
	else
	{
		cout << " --(!) No captured frame -- Break!" << endl;
		result = capture_image();
	}
	return result;
}


/** @function detectAndDisplay */
Mat detectFace(Mat& frame)
{
	cout << "detect face" << endl;
	VideoCapture cap(0);
	vector<Rect> faces;
	Mat frame_gray, cannyOut;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//imshow("Origin", faces);
	if (faces.empty() == false) {
		int size = 0;
		size_t location = 0;
		for (size_t l = 0; l < faces.size(); l++) {
			if (faces[l].width > size) {
				location = l;
				size = faces[l].width;
			}
		}
		cout << "goto eyelocation" << endl;
		Mat eyes = eyeLocation(frame, frame_gray, faces, location);
		if (eyes.empty()) {
			return eyes = capture_image();
		}
		else {
			return eyes;
		}
	}
	else {
		cout << "No eyes" << endl;
		return capture_image();
	}
	/*else {
	Mat rotated,rotatedFrame;
	Point2f centre;
	centre.x = frame.rows / 2;
	centre.y = frame.cols / 2;
	namedWindow("Control", WINDOW_NORMAL);
	for (int angle = 0; angle < 360; angle++) {
	warpAffine(frame,rotated,getRotationMatrix2D(centre, angle, 1),Size(frame.cols,frame.rows));
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	cout << angle << endl;
	angle = angle + 19;
	if (faces.empty() == false) {
	warpAffine(frame, rotatedFrame, getRotationMatrix2D(centre, angle, 1), Size(frame.cols, frame.rows));
	eyeLocation(rotatedFrame, rotated, faces);
	break;
	}
	imshow("Control", rotated);
	}
	}*/
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
Mat eyeLocation(Mat& frame, Mat& frame_gray, std::vector<Rect> faces, size_t location) {
	cout << "eye location" << endl;
	int eyeNormalX[80];
	int eyeNormalY[80];
	int eyeNormalRadius[80];
	int eyeRightX[80];
	int eyeRightY[80];
	int eyeRightRadius[80];
	int eyeLeftX[80];
	int eyeLeftY[80];
	int eyeLeftRadius[80];
	int mouthX[80];
	int mouthY[80];
	int mouthRadius[80];
	int averageHeight[80];
	int averageWidth[80];
	Mat frameCropped;
	size_t i = location;
	int o = 0;
	Point centerFace(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);


	Mat faceROI = frame_gray(faces[i]);
	vector<Rect> eyes_normal;
	vector<Rect> eyes_left;
	vector<Rect> eyes_right;
	vector<Rect> smile;

	//blue for normal
	eyes_cascade_normal.detectMultiScale(faceROI, eyes_normal, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t j = 0; j < eyes_normal.size(); j++)
	{
		eyeNormalX[j] = faces[i].x + eyes_normal[j].x + eyes_normal[j].width / 2;
		eyeNormalY[j] = faces[i].y + eyes_normal[j].y + eyes_normal[j].height / 2;
		eyeNormalRadius[j] = (eyes_normal[j].width + eyes_normal[j].height)*0.25;

		Point eye_center_normal(eyeNormalX[j], eyeNormalY[j]);
		int radius = cvRound(eyeNormalRadius[j]);

		//circle(frame, eye_center_normal, radius, Scalar(255, 0, 0), 4, 8, 0);


	}

	//green for left 
	eyes_cascade_left.detectMultiScale(faceROI, eyes_left, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t j = 0; j < eyes_left.size(); j++)
	{
		eyeLeftX[j] = faces[i].x + eyes_left[j].x + eyes_left[j].width / 2;
		eyeLeftY[j] = faces[i].y + eyes_left[j].y + eyes_left[j].height / 2;
		eyeLeftRadius[j] = (eyes_left[j].width + eyes_left[j].height)*0.25;

		Point eye_center_left(eyeLeftX[j], eyeLeftY[j]);
		int radius = cvRound(eyeLeftRadius[j]);
		//circle(frame, eye_center_left, radius, Scalar(0, 255, 0), 4, 8, 0);


	}

	//red for right
	eyes_cascade_right.detectMultiScale(faceROI, eyes_right, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t j = 0; j < eyes_right.size(); j++)
	{
		eyeRightX[j] = faces[i].x + eyes_right[j].x + eyes_right[j].width / 2;
		eyeRightY[j] = faces[i].y + eyes_right[j].y + eyes_right[j].height / 2;
		eyeRightRadius[j] = (eyes_right[j].width + eyes_right[j].height)*0.25;

		Point eye_center_right(eyeRightX[j], eyeRightY[j]);
		int radius = cvRound(eyeRightRadius[j]);
		//circle(frame, eye_center_right, radius, Scalar(0, 0, 255), 4, 8, 0);


	}

	//turquose for mouth
	smile_cascade.detectMultiScale(faceROI, smile, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t j = 0; j < smile.size(); j++)
	{
		mouthX[j] = faces[i].x + smile[j].x + smile[j].width / 2;
		mouthY[j] = faces[i].y + smile[j].y + smile[j].height / 2;
		mouthRadius[j] = (smile[j].width + smile[j].height)*0.25;

		Point mouth_center(mouthX[j], mouthY[j]);
		int radius = cvRound(mouthRadius[j]);
		//circle(frame, mouth_center, radius, Scalar(255, 255,0 ), 4, 8, 0);
		int p = j;

	}
	cout << "parts found" << endl;
	int eyesX[80];
	int eyesY[80];
	int eyesRadius[80];
	int l = 0;
	for (size_t h = 0; h < faces.size(); h++) {
		int correct = 0;
		for (size_t i = 0; i < eyes_normal.size(); i++) {
			for (size_t j = 0; j < eyes_left.size(); j++) {
				for (size_t k = 0; k < eyes_right.size(); k++) {
					if ((abs(eyeRightX[k] - eyeLeftX[j]) < eyeRightRadius[k] / 2) && (abs(eyeLeftY[j] - eyeRightY[k]) < eyeRightRadius[k] / 2)) {
						if ((abs(eyeLeftX[j] - eyeNormalX[i]) < eyeLeftRadius[j] / 2) && (abs(eyeNormalY[i] - eyeLeftY[j]) < eyeLeftRadius[j] / 2)) {
							if ((abs(eyeNormalX[i] - eyeRightX[k]) < eyeNormalRadius[i] / 2) && (abs(eyeRightY[k] - eyeNormalY[i]) < eyeNormalRadius[i] / 2)) {

								correct++;
								eyesX[l] = (eyeLeftX[j] + eyeRightX[k] + eyeNormalX[i]) / 3;
								eyesY[l] = ((eyeLeftY[j] + eyeRightY[k] + eyeNormalY[i]) / 3) + (faces[h].y)*0.05;
								eyesRadius[l] = (eyeLeftRadius[j] + eyeRightRadius[k] + eyeNormalRadius[i]) / 3;
								averageHeight[l] = (eyes_normal[i].height + eyes_left[j].height + eyes_right[k].height) / 3;
								averageWidth[l] = (eyes_normal[i].width + eyes_left[j].width + eyes_right[k].width) / 3;
								//cout << (faces[h].y) << endl;
								if ((abs(centerFace.x - eyesX[l]) < faces[o].width*0.5) && (abs(centerFace.y - eyesY[l]) < faces[o].width*0.5)) {
									//ellipse(frame, centerFace, Size(faces[o].width*0.5, faces[o].height*0.5), 0, 0, 180, Scalar(255, 0, 255), 4, 8, 0);
									l++;
								}
							}
						}
					}
				}
			}
		}
	}
	cout << "eye location found" << endl;
	int n;
	if ((eyesX[0] > 0) && (eyesX[1] > 0) && (eyesY[0] > 0) && (eyesY[1] > 0)) {
		Point eyes1(eyesX[0], eyesY[0]);
		Point eyes2(eyesX[1], eyesY[1]);
		//line(frame, eyes1, eyes2, Scalar(255, 255, 255), 5, 8, 0);
		cout << eyes1 << ", " << eyes2 << endl;
		if (eyesX[0] < eyesX[1]) {
			n = 0;
		}
		else {
			n = 1;
		}
		cout << "BOth eyes line drawn togeather" << endl;
		//Mat eyeROI1 = faceROI(eyes[p]);
		

		Rect cropped((eyesX[n] - (averageWidth[o] / 2)), (eyesY[n] - (averageHeight[o] / 2)), averageWidth[o], averageHeight[o]);
		frameCropped = frame(cropped);
		return frameCropped;
	}
	else {
		cout << "2 eyes not in range" << endl;
		return frameCropped = capture_image();
	}
}

void direction(Mat& frameCropped){
	Mat HSVImage, eyeWhites, greyImage, test, out, canny_output, out1, lab;
		cvtColor(frameCropped, greyImage, CV_RGB2GRAY);
		//GaussianBlur(greyImage, greyImage, Size(3,3), 2, 2);
		//inRange(HSVImage, Scalar(HL, SL, VL), Scalar(HH, SH, VH), eyeWhites);
		equalizeHist(greyImage, greyImage);
		equalizeHist(greyImage, greyImage);
		//equalizeHist(frameCropped, frameCropped);
		//Canny(out, cannyOut, 10, 75, 3);
		int erosion_size = 1;
		Mat element = getStructuringElement(2,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		dilate(greyImage, greyImage, element);
		erode(greyImage, greyImage, element);
		dilate(greyImage, greyImage, element);
		erode(greyImage, greyImage, element);
		//erode(greyImage, greyImage, element);
		//Mat eyeROI2 = faceROI(eyes[1]);



		//Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		/// Detect edges using canny

		Canny(greyImage, canny_output, 200, 300, 3);
		/// Find contours
		//findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Draw contours
		/*Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
		Scalar color = Scalar(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
		}
		//Canny(eyeROI1, cannyOut, 90, 180);
		*/
		//imshow(window_name2, eyeROI2);
		Mat framedOutput = frameCropped;
		vector<Vec3f> circles;
		//cout << "p" << endl;
		HoughCircles(canny_output, circles, CV_HOUGH_GRADIENT, 1, frameCropped.cols / 4, 250, 10, frameCropped.rows / 8, frameCropped.rows / 3);
		//resize(drawing, drawing, Size(300, 300), 2, 2);
		/// Draw the circles detected
		if (circles.size() == 0) { cout << "No circles found" << endl; }
		else {
			cout << circles.size() << endl;
			for (size_t i = 0; i < circles.size(); i++)
			{
				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]);
				// circle center
				//circle(framedOutput, center, 3, Scalar(0, 255, 0), -1, 8, 0);
				// circle outline
				//circle(framedOutput, center, radius, Scalar(0, 0, 255), 1, 8, 0);
			}
			cout << "Circles drawn" << endl;
		}

		//imshow("Origin", drawing);
		//resize(eyeWhites, eyeWhites, Size(300, 200), 2, 2);
		resize(frameCropped, frameCropped, Size(300, 300), 2, 2);
		resize(canny_output, canny_output, Size(300, 300), 2, 2);
		
		//resize(cannyOut, cannyOut, Size(300, 200), 2, 2);
		//addWeighted(out, 1, eyeWhites, .5, 0, test);
		imshow(window_name1, canny_output);

		
		imshow(window_name2, framedOutput);
		
	}




string get_time(void) {

		time_t rawtime;
		struct tm * timeinfo;
		char buffer[80];

		time(&rawtime);
		timeinfo = localtime(&rawtime);

		strftime(buffer, sizeof(buffer), "%d-%m-%Y_%I:%M:%S", timeinfo);
		std::string str(buffer);

		std::cout << str;

		return 0;
	
}

void training_image_left(Mat eye) {
	cout << "training looking left" << endl;
	resize(eye, eye, Size(300, 300), 2, 2);
	Mat greyImage,canny_output,HSV,eyeWhites,test;
	cvtColor(eye, greyImage, CV_RGB2GRAY);
	equalizeHist(greyImage, greyImage);
	equalizeHist(greyImage, greyImage);
	int erosion_size = 1;
	Mat element = getStructuringElement(2,Size(2 * erosion_size + 1, 2 * erosion_size + 1),Point(erosion_size, erosion_size));
	dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cvtColor(greyImage, greyImage, CV_GRAY2RGB);
	cvtColor(greyImage, HSV, CV_RGB2HSV);
	inRange(eye, Scalar(0,0,0,0), Scalar(180, 255, 30,0), eyeWhites);	
	Canny(greyImage, canny_output, 10, 80, 3);
	Mat framedOutput = eye;
	vector<Vec3f> circles;
	cout << "houghs circle" << endl;
	HoughCircles(canny_output, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 4, 250, 30, eye.rows / 8, eye.rows / 3);
	//resize(drawing, drawing, Size(300, 300), 2, 2);
	/// Draw the circles detected
	if (circles.size() == 0) { cout << "No circles found" << endl; }
	else {
		cout << circles.size() << endl;
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle(framedOutput, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(framedOutput, center, radius, Scalar(0, 0, 255), 1, 8, 0);
		}
		cout << "Circles drawn" << endl<<endl;
	}
//	addWeighted(greyImage, 1, eyeWhites, .5, 0, test);
	//resize(cannyOut, cannyOut, Size(300, 200), 2, 2);
	imshow(window_name1, greyImage);

}
void training_image_right(Mat eye) {
	resize(eye, eye, Size(300, 200), 2, 2);
}
void training_image_up(Mat eye) {
	resize(eye, eye, Size(300, 200), 2, 2);
}
void training_image_down(Mat eye) {
	resize(eye, eye, Size(300, 200), 2, 2);
}
void comparison() {

}