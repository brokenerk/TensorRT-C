#include "Camara.h"

string __VIDEO_PATH = "./../tests/";
string video = "videoSSD.mp4";

Camara::Camara(int tipoCamara) {
	this->tipoCamara = tipoCamara;
	int __width = 1280;
	int __height = 720;

	switch(this->tipoCamara) {
		case 0: {
			string __user = "admin";
			string __password = "Uno+dos3";
			string __ip = "169.254.47.87";
			string __uri = "rtsp://" + __user + ":" + __password + "@" + __ip;
			int __latency = 160;
			
			string gst_str = "rtspsrc location="+__uri+" latency="+to_string(__latency)+" ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)"+to_string(__width)+", height=(int)"+to_string(__height)+", format=(string)BGRx ! videoconvert ! appsink";
			this->__cap = cv::VideoCapture(gst_str, cv::CAP_GSTREAMER);
		}
		break;
		case 1: {
	        int __fps = 22;
	        this->__cap = cv::VideoCapture("/dev/video1");
	        this->__cap.set(cv::CAP_PROP_FRAME_WIDTH, __width);
	        this->__cap.set(cv::CAP_PROP_FRAME_HEIGHT, __height);
	        this->__cap.set(cv::CAP_PROP_FPS, __fps);

		}
		break;
		case 2: {
			this->__cap = cv::VideoCapture(__VIDEO_PATH+video);
		}
		break;
	}
}

cv::Mat Camara::getFrame() {
	cv::Mat img;
	bool validated = this->__cap.read(img);

	switch(this->tipoCamara) {
		case 0:
			return img;
		break;
		case 1:{
			cv::flip(img, img, 1);
			return img;
		}
		break;
		case 2: {
			if(!validated) {
				this->__cap = cv::VideoCapture(__VIDEO_PATH+video);
				bool validated = this->__cap.read(img);
			}
			return img;
		}
		break;	
	}
}

void Camara::desconectar() {
	this->__cap.release();
}