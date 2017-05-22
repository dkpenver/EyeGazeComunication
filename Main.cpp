#include<opencv2\opencv.hpp>
#include "SkinDetector.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

int main()
{
	Mat src; Mat src_gray;
	int thresh = 120;
	int max_thresh = 255;
	RNG rng(12345);
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)

	capture.open(1);

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1080);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1920);

	Mat cameraFeed,cannyOut;

	SkinDetector mySkinDetector;

	Mat skinMat,hist;

	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while (1) {

		//store image to matrix
		capture.read(cameraFeed);

		//show the current image
		//imshow("Original Image", cameraFeed);

		skinMat = mySkinDetector.getSkin(cameraFeed);

		imshow("Skin Image", skinMat);
		cvtColor(cameraFeed, cameraFeed, CV_BGR2GRAY);
		////////////////////////////////////////////////////////////////////////////////////////////
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		/// Detect edges using canny
		Canny(cameraFeed, canny_output, thresh, thresh * 2, 3);
		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		imshow("Original Image", drawing);
		if (waitKey(30) == 32) { break; };
	
	}
	return 0;
}