#include "VideoPredefinido.h"

VideoPredefinido::VideoPredefinido() {
	__VIDEO_PATH = string("./../tests/");
	__cap = cv::VideoCapture(__VIDEO_PATH+video);
}

cv::Mat VideoPredefinido::getFrame() {
	cv::Mat img;
	bool validated = __cap.read(img);
	if(!validated) {
		__cap = cv::VideoCapture(__VIDEO_PATH+video);
		bool validated = __cap.read(img);
	}
	return img;
}

void VideoPredefinido::desconectar() {
	__cap.release();
}