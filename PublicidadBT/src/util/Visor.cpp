#include "Visor.h"

Visor::Visor(int tipoCamara) {
	this->__camara = new Camara(tipoCamara);

	cv::namedWindow(this->__nombreVentana,cv::WINDOW_NORMAL);
	cv::resizeWindow(this->__nombreVentana, 1280, 720);
	this->iniciar();
}

void Visor::redEG(cv::Mat rostro, float* e_g) {
	this->__cnnEG.obtenerEG(rostro, e_g);
}

void Visor::redBig5(cv::Mat rostro, float* big5) {
	cv::Mat rostro_gray;
	cv::cvtColor(rostro, rostro_gray, cv::COLOR_RGB2GRAY);
	this->__cnnBig5.obtenerPersonalidad(rostro_gray, big5);
}

void Visor::procesarRostroNuevo(Rostro rostro) {
	float personalidad[5];
	float e_g[2];
	cv::Mat rostro_img = rostro.getImg();

	this->redEG(rostro_img, e_g);
	this->redBig5(rostro_img, personalidad);

	rostro.setPersonalidad(personalidad);
	rostro.setEdadGenero(e_g);

	cout << "\nEdad: " << e_g[0] << " Genero: " << e_g[1] << endl;
	cout << "Big5: [" << personalidad[0] << " " << personalidad[1] << " " << personalidad[3] << " " << personalidad[4] << "]" << endl;
}

bool Visor::compararRostros(Rostro rostroA, Rostro rostroN) {
	float tolerancia = 0.2;
	int bba[4];
	int bbn[4];
	rostroA.getCoordenadas(bba);
	rostroN.getCoordenadas(bbn);
	int e1 = abs(bba[0]-bbn[0])/(bba[2]- bba[0]);
	int e2 = abs(bba[2]-bbn[2])/(bba[2]- bba[0]);
	int e3 = abs(bba[1]-bbn[1])/(bba[3]- bba[1]);
	int e4 = abs(bba[3]-bbn[3])/(bba[3]- bba[1]);
	return ((e1+e2+e3+e4) / 4) < tolerancia;
}

void Visor::actualizarRostros(vector<Rostro> rostros) {
	int frames_maximos = 4;
	// Recorre todos los rostros que fueron detectados anteriormente
	vector<Rostro> rostros_a_eliminar;
	for(auto &r : this->__rostros) {
		Rostro rostroA = r.first;
		bool coincidencia = false;
		// Recorre todos los rostros nuevos que detecto la SSD
		for(Rostro rostroN : rostros) {
			// Si al comparar la ubicacion de los rostros se retorna menos de 15, entonces el rostro nuevo es el mismo que el anterior
			if(this->compararRostros(rostroA, rostroN)) {
				// Actualizamos coordenadas
				int rostroA_coordenadas[4];
				rostroN.getCoordenadas(rostroA_coordenadas);
				rostroA.setCoordenadas(rostroA_coordenadas);
				// Actualizamos el contador 
				this->__rostros[rostroA] = frames_maximos;
				// Bandera para saber que hubo coincidencia
				coincidencia = true;
				break;
			}	
		}
		// Tras revisar todos los rostros detectados por la SSD, si no hubo coincidencia el rostro podria ya no estar
		if(!coincidencia) {
			// Obtenemos el contador. Este contador sirve para saber por cuantos frames mas se puede intentar detectar
			int contadorRostro = this->__rostros[rostroA];
			// Si el contador es mayor a 0 solo se decrementa
			if(contadorRostro > 0)
				this->__rostros[rostroA] = contadorRostro - 1;
			else
				rostros_a_eliminar.push_back(rostroA);
			// En caso contrario, quiere decir que el rostro no ha sido detectado en 5 veces, por lo que se elimina de los rostros que seran usados para la RA
		}
	}

	for(Rostro rostro : rostros_a_eliminar)
		this->__rostros.erase(rostro);
	//	Si quedan rostros  en la lista recibida despues de compararlos con todos los rostros que ya existian, 
	// entonces es un rostro nuevo al que hay que recomendar algo y agregar al diccionario de rostros
	for(Rostro rostroN : rostros) {
		this->procesarRostroNuevo(rostroN);
		this->__rostros[rostroN] = frames_maximos;
	}
	if(rostros.size() > 0)
		this->__nuevaRecomendacion = true;
	else
		this->__nuevaRecomendacion = false;
}

cv::Mat Visor::procesar(cv::Mat imgC) {
	// La SSD detecta los rostros de la imagen
	vector<Rostro> rostros = this->__ssd.detectarRostros(imgC);
	// Convertimos imagen en BGRA para trabajar con alpha
	cv::cvtColor(imgC, imgC, cv::COLOR_BGR2BGRA);
	// Verifica si cada rostro coincide con un rostro detectado anteriormente.
	this->actualizarRostros(rostros);
	cv::Mat img;
	if(rostros.size() > 0) {
		img = imgC;
	}
	else {
		img = cv::imread("./../tests/static2.jpg");
	}
	return img;
}

void Visor::iniciar() {
	bool full_scrn = false;
	while(true) {
		// Se recupera en frame desde la camara
		cv::Mat imgC = this->__camara->getFrame();
		cv::Mat img = this->procesar(imgC);
		cv::imshow(this->__nombreVentana, img);

		char key = cv::waitKey(10);
        if (key == 81 or key == 113) //Metodo para salir, oprimir la letra Q
            break;
        else if (key == 70 or key == 102) {
        	// Pantalla Completa, oprimir la tecla F
        	full_scrn = !full_scrn;
        	if(full_scrn)
        		cv::setWindowProperty(this->__nombreVentana, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        	else
        		cv::setWindowProperty(this->__nombreVentana, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
        }
	}
	this->finalizar();
}

void Visor::finalizar() {
	this->__camara->desconectar();
	cv::destroyAllWindows();
}