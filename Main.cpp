#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
VideoCapture video;
int sliderPosition = 0;
int run = 1, step = 0;

int main(int agrc, char * argc[]) {
	video.open(1);
	namedWindow(" Video ", WINDOW_AUTOSIZE);
	Mat frame;
	while (true) {
		if (run | step) {
			video >> frame;
			if (frame.empty()) break;
			imshow(" Video ", frame);
		}
		char c = waitKey(33);
		if (c == 27) break;
	}
	system(" pause ");
	return 0;
}