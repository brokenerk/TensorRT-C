#ifndef CAMARA_H_
#define CAMARA_H_
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
class Camara {
	public:
	    Camara(int);
	    cv::Mat getFrame();
	    void desconectar();
	private:
	    cv::VideoCapture __cap;
	    int tipoCamara;
};
#endif