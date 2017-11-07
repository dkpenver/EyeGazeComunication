#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <array>
#include <ctime>
#include <string>
#include <fstream>


#define MAX_DATE 12

using namespace std;
using namespace cv;
//using namespace dlib;

Mat cameraFeed;
Mat src; 
Mat src_gray;
int thresh = 25;
int max_thresh = 255;
RNG rng(12345);
Mat frame,image;

/** Function Headers */
Mat detectFace(Mat& frame);
Mat eyeLocation(Mat& frame, Mat& frame_grey, std::vector<Rect> faces, size_t location);
string get_time(void);
Mat eyeFiltering(Mat eye);
Mat capture_image();
void comparison(Mat image);
Mat averageingImages(Mat image1,Mat image2,float count);
int counter(void);
float reset(float count);
Mat filter(Mat image,float count);
float accuracy(Mat trained, Mat check);



/** Global variables */
int HSVValue = 0;
int threshold_value = 0;
int threshold_type = 1;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
Mat	 dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

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
std::string upMerged= "Up";
std::string downMerged ="Down";
std::string leftMerged = "Left";
std::string rightMerged = "Right";
std::string window_name5 = "Capture5 - Face detection";



int main()
{

	face_cascade.load(face_cascade_name);
	eyes_cascade_normal.load(eyes_cascade_normal_name);
	eyes_cascade_left.load(eyes_cascade_left_name);
	eyes_cascade_right.load(eyes_cascade_right_name);
	smile_cascade.load(smile_cascade_name);
	//VideoCapture::VideoCapture(1);

	//namedWindow("Origin", WINDOW_AUTOSIZE);

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade_normal.load(eyes_cascade_normal_name)) { printf("--(!)Error loading\n"); return -1; };

	Mat left, right, up, down, loop,average,filtered;
	char input;

	cout << "Do you want to train (Input 't') or do you want to detect(Input 'l') " << endl;
	cin >> input;

	if (input == 't') {
		float count = counter();
		count = reset(count);
		while (1) {
			float count = counter();
			cout << "look left" << endl;
			waitKey(0);
			left = capture_image();
			image = eyeFiltering(left);
			if (count == 0) {
				average = image;
				//cout << "count = 0" << endl;
			}
			else {
				left = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/left.jpg", 0);
				average = averageingImages(image, left, count);
				//cout << "count != 0" << endl;
			}
			filtered = filter(average, count);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/leftFilter.jpg", filtered);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/left.jpg", average);
			imshow("Origin", filtered);
			cout << endl;


			cout << "Look right" << endl;
			waitKey(0);
			right = capture_image();
			image = eyeFiltering(right);
			if (count == 0) {
				average = image;
			}
			else {
				right = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/right.jpg", 0);
				average = averageingImages(image, right, count);
			}
			filtered = filter(average, count);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/rightFilter.jpg", filtered);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/right.jpg", average);
			imshow("Origin", filtered);
			cout << endl;


			cout << "Look up" << endl;
			waitKey(0);
			up = capture_image();
			image = eyeFiltering(up);
			if (count == 0) {
				average = image;
			}
			else {
				up = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/up.jpg", 0);
				average = averageingImages(image, up, count);
			}
			filtered = filter(average, count);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/upFilter.jpg", filtered);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/up.jpg", average);
			imshow("Origin", filtered);
			cout << endl;


			cout << "Look down" << endl;
			waitKey(0);
			down = capture_image();
			image = eyeFiltering(down);
			if (count == 0) {
				average = image;
			}
			else {
				down = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/down.jpg", 0);
				average = averageingImages(image, down, count);
			}
			filtered = filter(average, count);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/downFilter.jpg", filtered);
			imwrite("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/down.jpg", average);
			imshow("Origin", filtered);
			waitKey(1);
			if (count == 0) { counter(); }
			cout << endl;
			cout << "Continue" << endl;
			cin >> input;
			if (input == 'n') {
				break;
			}
		}
	
	}

	if (input == 'l') {
		cout << "loop begins" << endl << endl;
		while (true) {
			loop = capture_image();
			image = eyeFiltering(loop);
			comparison(image);
			imshow("Origin", loop);
			waitKey(0);
			cout << endl;
		}
	}
}

Mat capture_image() {
	cout << "capture image" << endl;
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	Mat frame,result, input;
	cap.read(frame);
	resize(frame, input, Size(300, 200), 2, 2);
	//imshow("window_name1", input);
	//waitKey(1);
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
		//cout << "goto eyelocation" << endl;
		Mat eyes = eyeLocation(frame, frame_gray, faces, location);
		if (eyes.empty()) {
			return eyes = capture_image();
		}
		else {
			return eyes;
		}
	}
	else {
		//cout << "No eyes" << endl;

		return capture_image();
	}
}
Mat eyeLocation(Mat& frame, Mat& frame_gray, std::vector<Rect> faces, size_t location) {
	//cout << "eye location" << endl;
	int eyeNormalX[80];
	int eyeNormalY[80];
	int eyeNormalRadius[80];
	int eyeRightX[80];
	int eyeRightY[80];
	int eyeRightRadius[80];
	int eyeLeftX[80];
	int eyeLeftY[80];
	int eyeLeftRadius[80];
	//int mouthX[80];
	//int mouthY[80];
	//int mouthRadius[80];
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

		circle(frame, eye_center_normal, radius, Scalar(255, 0, 0), 4, 8, 0);


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
		circle(frame, eye_center_left, radius, Scalar(0, 255, 0), 4, 8, 0);


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
		circle(frame, eye_center_right, radius, Scalar(0, 0, 255), 4, 8, 0);


	}
	resize(frame, frame, Size(1000, 600), 2, 2);
	imshow(window_name,frame);
	waitKey(0);
	//turquose for mouth
	/*smile_cascade.detectMultiScale(faceROI, smile, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t j = 0; j < smile.size(); j++)
	{
		mouthX[j] = faces[i].x + smile[j].x + smile[j].width / 2;
		mouthY[j] = faces[i].y + smile[j].y + smile[j].height / 2;
		mouthRadius[j] = (smile[j].width + smile[j].height)*0.25;
		Point mouth_center(mouthX[j], mouthY[j]);
		int radius = cvRound(mouthRadius[j]);
		//circle(frame, mouth_center, radius, Scalar(255, 255,0 ), 4, 8, 0);
		int p = j;
	}*/
	//cout << "parts found" << endl;
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
	//cout << "eye location found" << endl;
	int n;
	if ((eyesX[0] > 0) && (eyesX[1] > 0) && (eyesY[0] > 0) && (eyesY[1] > 0)) {
		Point eyes1(eyesX[0], eyesY[0]);
		Point eyes2(eyesX[1], eyesY[1]);
		//line(frame, eyes1, eyes2, Scalar(255, 255, 255), 5, 8, 0);
		cout << eyes1 << ", " << eyes2 << endl;
		if ((eyesX[0] < eyesX[1])|| (eyesY[0] < eyesY[1])) {
			n = 0;
		}
		else {
			n = 1;
		}
		//cout << "BOth eyes line drawn togeather" << endl;
		//Mat eyeROI1 = faceROI(eyes[p]);
		

		Rect cropped((eyesX[n] - (averageWidth[o] / 2)), (eyesY[n] - (averageHeight[o] / 2)), averageWidth[o], averageHeight[o]);
		frameCropped = frame(cropped);
		return frameCropped;
	}
	else {
		if ((eyesX[0]> 0) && (eyesY > 0)){
			Point eyes1(eyesX[0], eyesY[0]);
			Rect cropped((eyesX[0] - (averageWidth[o] / 2)), (eyesY[0] - (averageHeight[o] / 2)), averageWidth[o], averageHeight[o]);
			frameCropped = frame(cropped);
			return frameCropped;
		}
		//cout << "2 eyes not in range" << endl;
		return frameCropped = capture_image();
	}
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

Mat eyeFiltering(Mat eye) {
	//cout << "training looking left" << endl;
	
	Mat greyImage,canny_output,HSV,eyeWhites,test,output;
	cvtColor(eye, greyImage, CV_RGB2GRAY);
	equalizeHist(greyImage, greyImage);
	equalizeHist(greyImage, greyImage);
	int erosion_size = 1;
	Mat element = getStructuringElement(2,Size(2 * erosion_size + 1, 2 * erosion_size + 1),Point(erosion_size, erosion_size));
	/*dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	dilate(greyImage, greyImage, element);
	erode(greyImage, greyImage, element);
	*/
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cvtColor(greyImage, greyImage, CV_GRAY2RGB);
	cvtColor(greyImage, HSV, CV_RGB2HSV);
	for (int num = 0; num < 5000; HSVValue++) {
		inRange(eye, Scalar(0, 0, 0, 0), Scalar(180, 255, HSVValue, 0), eyeWhites);
		for (int y = 0; y < greyImage.rows; y++) {
			for (int x = 0; x < greyImage.cols; x++) {
				if ((eyeWhites.at<uchar>(y, x) > 100)) {
					num++;
				}
			}
		}
		//cout <<HSVValue<< "HSV "<<num << endl;
	}
	HSVValue = 0;
	Canny(greyImage, canny_output, 10, 600, 3);
	Mat framedOutput = eye;
	vector<Vec3f> circles;
	//cout << "houghs circle" << endl;
	
	src_gray = greyImage;

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold

	/// Call the function to initialize

	for (int num = 0; num < 8000;threshold_value++) {
		threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);
		for (int y = 0; y < src_gray.rows; y++) {
			for (int x = 0; x < src_gray.cols; x++) {
				if ((dst.at<uchar>(y, x) >100)) {
					num++;
				}
			}
		}
		//cout <<threshold_value<<"binary "<< num << endl;
	}
	threshold_value = 0;
	//cout << threshold_value << endl;

	cvtColor(dst, dst, CV_RGB2BGR);
	cvtColor(eyeWhites, canny_output, CV_GRAY2RGB);
	resize(dst, dst, Size(600, 600), 2, 2);
	resize(canny_output, canny_output, Size(600, 600), 2, 2);
	//cout << "resized" << endl; 
	addWeighted(dst, 1, canny_output, 1, 0, test);
	
	//cout << "weighted" << endl;
	cvtColor(test,test, CV_RGB2GRAY);
	
	
	

	/*HoughCircles(test, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 4, 250, 30, eye.rows / 8, eye.rows / 2);
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
		cout << "Circles drawn" << endl << endl;
	}*/
	//resize(cannyOut, cannyOut, Size(300, 200), 2,2);
	//resize(canny_output, output, Size(300, 300), 2, 2);
	//imshow(window_name1, output);
	//resize(dst, dst, Size(600, 600), 2, 2);
	imshow(window_name, test);
	return test;
}

void comparison(Mat image) {
	Mat up, down, left, right, mergedLeft,mergedRight,mergedUp,mergedDown;
	left = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/leftFilter.jpg", 1);
	right = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/rightFilter.jpg", 1);
	up = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/upFilter.jpg", 1);
	down = imread("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/downFilter.jpg", 1);
	
	resize(left, left, Size(300, 300), 2, 2);
	
	cvtColor(left,left , CV_BGR2GRAY);

	resize(right, right, Size(300, 300), 2, 2);
	cvtColor(right,right , CV_BGR2GRAY);
	resize(up, up, Size(300, 300), 2, 2);
	cvtColor(up,up , CV_BGR2GRAY);
	resize(down,down, Size(300, 300), 2, 2);
	cvtColor(down,down , CV_BGR2GRAY);
	resize(image, image, Size(300, 300), 2, 2);
	//cout << "i" << endl;

	float leftAccuracy, rightAccuracy, upAccuracy, downAccuracy;
	leftAccuracy= accuracy(left,image);
	rightAccuracy = accuracy(right, image);
	upAccuracy = accuracy(up, image);
	downAccuracy = accuracy(down, image);
	cout << leftAccuracy << "% Left" <<endl<< rightAccuracy << "% Right" <<endl<< upAccuracy << "% Up" <<endl<< downAccuracy << "% Down" << endl;
	if ((upAccuracy > (2 * leftAccuracy)) && (upAccuracy > (2 * rightAccuracy))) {
		cout << "Looking  up" << endl;
	}
	else if ((downAccuracy/upAccuracy)> 2) {
		cout << "Looking  Down" << endl;
	}
	else if (((leftAccuracy / upAccuracy) > 0.5)&&((rightAccuracy/upAccuracy)>0.5)) {
		if (leftAccuracy > rightAccuracy) {
			cout << "Looking  left" << endl;
		}
		else {
			cout << "Looking Right" << endl;

		}

	}
	else {
		cout << "Error" << endl;
	}
	
	
	
	//imshow(rightMerged, image);
}
Mat averageingImages(Mat image2,Mat image1,float count) {
	Mat averaged= Mat(image1.rows, image1.cols, CV_32F);
	cout << 1 - (1 / count) <<","<< 1/count<<","<<count<< endl;
	addWeighted(image1, (1-(1/count)), image2, (1/count), 0, averaged);
	
return averaged;
}
Mat filter(Mat averaged, float count) {
	int count_black = 0;
	int count_white = 0;
	cout << "filter" << endl;
	Mat filtered = Mat(averaged.rows, averaged.cols, CV_32F);
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < averaged.cols; x++) {
			if (averaged.at<uchar>(y, x) > 170) {
				// change this to to 'src.atuchar>(y,x) == 255' 
				// if your img has only 1 channel
				//cout << "add white" << endl;
				filtered.at<float>(y, x) = 255;
				count_black++;
			}
		}

	}
	int erosion_size = 1;
	Mat element = getStructuringElement(2, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	cout << count_black << endl;
	dilate(filtered, filtered, element);
	dilate(filtered, filtered, element);
	erode(filtered, filtered, element);
	erode(filtered, filtered, element);
	return filtered;
}
int counter(void) {
	string line;
	ifstream myfile("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/counter.txt");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';
		}
		myfile.close();
	}
	else cout << "Unable to open file";

	int count = stoi(line.c_str());
	count = count + 1;

	ofstream myfileIn("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/counter.txt");
	if (myfileIn.is_open())
	{
		myfileIn << count;
		myfileIn.close();
	}
	else cout << "Unable to open file";

	return count;
}
float reset(float count) {
	char input;
	cout << "Count: " << count << endl << "Do you want to reset y/n" << endl;
	cin >> input;
	if (input == 'y') {
		string line;
		cout << "Counter reset" << endl;
		int count;
		count = -1;

		ofstream myfileIn("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/counter.txt");
		if (myfileIn.is_open())
		{
			myfileIn << count;
			myfileIn.close();
		}
		else cout << "Unable to open file";

	}
	else { cout << "Counter not reset" << endl; }
	return count;
}

float accuracy(Mat trained, Mat check) {
	Mat filtered = Mat(trained.rows, trained.cols, CV_32F);
	for (int y = 0; y < trained.rows; y++) {
		for (int x = 0; x < trained.cols; x++) {
			if (check.at<uchar>(y, x) > 10) {
				filtered.at<float>(y, x) = 255;
			}
			else {
				filtered.at<float>(y, x) = 0;
			}
		}
	}

	//cout << "new Mat made" << endl;
	float sameWhitePixels = 0;
	float totalWhite = 0;
	float excessWhite = 0;
	float sameColour = 0;
	float whitePixelsTrained = 0;
	float whitePixelsFiltered = 0;
	//imshow(leftMerged, filtered);
	for (int y = 0; y < trained.rows; y++) {
		for (int x = 0; x < trained.cols; x++) {
			if ((filtered.at<float>(y, x) == 255) ^ ((trained.at<uchar>(y, x)) == 255)) {
				excessWhite++;
			}
			if ((filtered.at<float>(y, x) == 255) || (trained.at<uchar>(y, x)) == 255) {
				totalWhite++;
			}
			//cout << "total pixels counted" << endl;

			//cout << filtered.at<float>(y, x) <<","<<colour<< endl;
			if (((filtered.at<float>(y, x)) ==255)&& ((trained.at<uchar>(y, x))==255)){
				//cout << "filtered" << endl;
				sameWhitePixels++;
			}
			if ((trained.at<uchar>(y, x)) == 255) {
				//cout << "filtered" << endl;
				whitePixelsTrained++;
			}
			if (((filtered.at<float>(y, x)) == 255)) {
				//cout << "filtered" << endl;
				whitePixelsFiltered++;
			}
			if (filtered.at<float>(y, x) ==  (trained.at<uchar>(y, x))) {
				//cout << "filtered" << endl;
				sameColour++;
			}
		}
	}
	ofstream myfileIn;
	myfileIn.open("C:/Users/Kurt/Documents/Visual Studio 2015/Projects/EyeGazeComunication/images/data.txt", ios::app);
	if (myfileIn.is_open())
	{
		myfileIn << totalWhite << endl << sameWhitePixels << endl << excessWhite << endl <<endl;
		myfileIn.close();
	}
	else cout << "Unable to open file";

	//waitKey(0);
	cout << totalWhite << "," << sameWhitePixels << "," << excessWhite << "," << sameColour << "," << whitePixelsFiltered << "," << whitePixelsTrained << endl;
	float output = ((sameWhitePixels / totalWhite)*100*(totalWhite/excessWhite)*(sameColour/100000));
	//cout << output << endl;
	return output;
}