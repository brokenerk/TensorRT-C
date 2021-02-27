#include "camera/VideoPredefinido.h"

int main() {
	VideoPredefinido video;
	cv::namedWindow("Video Demo", cv::WINDOW_AUTOSIZE);

	while(1) {
		cv::Mat imgC = video.getFrame();
		cv::imshow("Video Demo", imgC);
		char key = cv::waitKey(10);
        if (key == 81 or key == 113) //Metodo para salir, oprimir la letra Q
            break;
	}
	video.desconectar();
    return 0;
}