#include "camera/VideoPredefinido.h"
#include "rna/DetectorRostros.h"
#include "rna/CnnEG.h"
#include "rna/CnnBig5.h"

int main() {
	VideoPredefinido camara;
	DetectorRostros ssd;
	CnnEG cnnEG;
	CnnBig5 cnnBig5;

	string __nombreVentana = "BubbleTown";
	cv::namedWindow(__nombreVentana, cv::WINDOW_NORMAL);
	cv::resizeWindow(__nombreVentana, 208, 208);

	while(1) {
		cv::Mat imgC = camara.getFrame();
		vector<cv::Mat> rostros = ssd.detectarRostros(imgC);

		for(cv::Mat rostroN : rostros){
			cv::imshow(__nombreVentana, rostroN);

			float e_g[2];
			float big5[5];
			cnnEG.obtenerEG(rostroN, e_g);

			cv::Mat rostro_gray;
			cv::cvtColor(rostroN, rostro_gray, cv::COLOR_RGB2GRAY);
			cnnBig5.obtenerPersonalidad(rostro_gray, big5);

			cout << "Edad: " << e_g[0] << " Genero: " << e_g[1] << endl;
			cout << "Big5: [" << big5[0] << " " << big5[1] << " " << big5[3] << " " << big5[4] << "]" << endl;
		}
		char key = cv::waitKey(10);
        if (key == 81 or key == 113) //Metodo para salir, oprimir la letra Q
            break;
		
	}
	camara.desconectar();
    return 0;
}