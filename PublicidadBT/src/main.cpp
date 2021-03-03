#include "util/Visor.h"

int main(int argc, char** argv) {
	if (argc != 2) {
        printf("usage: main tipoCamara\n");
        return -1;
    }

	int tipoCamara = atoi(argv[1]);
	Visor visor(tipoCamara);
	return 0;
}