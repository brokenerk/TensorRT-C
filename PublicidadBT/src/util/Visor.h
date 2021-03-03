#ifndef VISOR_H_
#define VISOR_H_
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <iterator>
#include <cstdlib> 
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "./../camara/Camara.h"
#include "./../rna/DetectorRostros.h"
#include "./../rna/CnnEG.h"
#include "./../rna/CnnBig5.h"
#include "./../persistencia/Rostro.h"
using namespace std;
class Visor {
	public:
	    Visor(int);
	    void redEG(cv::Mat rostro, float* e_g);
	    void redBig5(cv::Mat rostro, float* big5);
	    void procesarRostroNuevo(Rostro rostro);
	    bool compararRostros(Rostro rostroA, Rostro rostroN);
	    void actualizarRostros(vector<Rostro> rostros);
	    cv::Mat procesar(cv::Mat imgC);
	    void iniciar();
	    void finalizar();
	private:
	    CnnEG __cnnEG;
	    CnnBig5 __cnnBig5;
	    string __nombreVentana = "BubbleTown";
	    DetectorRostros __ssd;
	    int __rostrosAnt = 0;
	    bool __nuevaRecomendacion = false;
	    map<Rostro, int> __rostros;
	    Camara* __camara;
};
#endif