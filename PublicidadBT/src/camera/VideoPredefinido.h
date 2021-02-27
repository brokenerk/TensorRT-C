#ifndef VIDEOPREFEDINIDO_H_
#define VIDEOPREDEFINIDO_H_
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
class VideoPredefinido {
	string video = "videoSSD.mp4";
	public:
	    VideoPredefinido();
	    cv::Mat getFrame();
	    void desconectar();
	private:
	    string __VIDEO_PATH;
	    cv::VideoCapture __cap;
};
#endif