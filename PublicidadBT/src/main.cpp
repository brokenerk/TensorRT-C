#include "camera/VideoPredefinido.h"
#include "rna/DetectorRostros.h"

int main() {
	VideoPredefinido camara;
	DetectorRostros ssd;

	string __nombreVentana = "BubbleTown";
	cv::namedWindow(__nombreVentana, cv::WINDOW_NORMAL);
	cv::resizeWindow(__nombreVentana, 208, 208);

	while(1) {
		cv::Mat imgC = camara.getFrame();
		vector<cv::Mat> rostros = ssd.detectarRostros(imgC);

		for(cv::Mat r : rostros){
			cv::imshow(__nombreVentana, r);
		}
		char key = cv::waitKey(10);
        if (key == 81 or key == 113) //Metodo para salir, oprimir la letra Q
            break;
		
	}
	camara.desconectar();
    return 0;
}