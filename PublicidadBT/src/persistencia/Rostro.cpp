#include "Rostro.h"

Rostro::Rostro(int id) {
	if(id != NULL) {
		this->__id = id;
	}
}
// GETTERS
int Rostro::getId() {
	return this->__id;
}
cv::Mat Rostro::getImg() {
	return this->__img;
}
int Rostro::getPosicionX() {
	return this->__posicionX;
}
int Rostro::getPosicionY() {
	return this->__posicionY;
}
int Rostro::getAlto() {
	return this->__alto;
}
int Rostro::getAncho() {
	return this->__ancho;
}
int Rostro::getGenero() {
	return this->__genero;
}
float Rostro::getEdad() {
	return this->__edad;
}
void Rostro::getPersonalidad(float* big5) {
	for(int i = 0; i < 5; i++)
		big5[i] = this->__personalidad[i];
}
void Rostro::getCoordenadas(int* coordenadas) {
	for(int i = 0; i < 4; i++)
		coordenadas[i] = this->__coordenadas[i];
}

// SETTERS
void Rostro::setImg(cv::Mat img) {
	this->__img = img;
}
void Rostro::setPosicionX(int posicionX) {
	this->__posicionX = posicionX;
}
void Rostro::setPosicionY(int posicionY) {
	this->__posicionY = posicionY;
}
void Rostro::setAlto(int alto) {
	this->__alto = alto;
}
void Rostro::setAncho(int ancho) {
	this->__ancho = ancho;
}
void Rostro::setGenero(int genero) {
	this->__genero = genero;
}
void Rostro::setEdad(float edad) {
	this->__edad = edad;
}
void Rostro::setEdadGenero(float* edad_genero){
	this->__edad = edad_genero[0];
	this->__genero = (int)edad_genero[1];
}
void Rostro::setPersonalidad(float* big5) {
	for(int i = 0; i < 5; i++)
		this->__personalidad[i] = big5[i];
}
void Rostro::setCoordenadas(int* coordenadas) {
	for(int i = 0; i < 4; i++)
		this->__coordenadas[i] = coordenadas[i];
}