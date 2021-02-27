OPENCV = `pkg-config --libs opencv4 --cflags opencv4`

run: main.cpp VideoPredefinido.o
	g++ main.cpp VideoPredefinido.o $(OPENCV) -o ./../bin/main
	
VideoPredefinido.o: ./camera/VideoPredefinido.cpp
	g++ ./camera/VideoPredefinido.cpp $(OPENCV) -c 

clean:
	rm -r *.o