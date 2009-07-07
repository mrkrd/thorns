all : _cThorns.so

_cThorns.o : _cThorns.c
	gcc -fPIC `python-config --cflags` \
	-I `python -c "import numpy; print numpy.get_include()"` \
	-c _cThorns.c -o _cThorns.o

_cThorns.so : _cThorns.o
	gcc -shared `python-config --ldflags` -o _cThorns.so _cThorns.o

clean :
	rm -f _cThorns.so _cThorns.o
