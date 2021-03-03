#ifndef ROSTRO_H_
#define ROSTRO_H_
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <iterator>
#include <map>
#include <opencv2/opencv.hpp>
using namespace std;
class Rostro {
	public:
	    Rostro(int id=NULL);
	    int getId();
	    cv::Mat getImg();
	    int getPosicionX();
	    int getPosicionY();
	    int getAlto();
	    int getAncho();
	    int getGenero();
	    float getEdad();
	    void getPersonalidad(float* big5);
	    void getCoordenadas(int* coordenadas);

	    void setImg(cv::Mat img);
	    void setPosicionX(int posicionX);
	    void setPosicionY(int posicionY);
	    void setAlto(int alto);
	    void setAncho(int ancho);
	    void setGenero(int genero);
	    void setEdad(float edad);
	    void setEdadGenero(float* edad_genero);
	    void setPersonalidad(float* big5);
	    void setCoordenadas(int* coordenadas);

	    bool operator <(const Rostro& r) const {
	        return __id < r.__id;
	    }
	private:
	    int __id = 0;
	    cv::Mat __img;
	    int __posicionX;
	    int __posicionY;
	    int __alto;
	    int __ancho;
	    int __genero;
	    float __edad;
	    float __personalidad[5];
	    float __coordenadas[4];
};
#endif