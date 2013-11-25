all: clogistic.so

clogistic.c: clogistic.pyx
	cython clogistic.pyx

clogistic.so: clogistic.c
	python -c 'import pyximport; pyximport.install(inplace=True); import clogistic'
#	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/numpy -lpython2.7 -o clogistic.so clogistic.c
#	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -lpython2.7 -o clogistic.so clogistic.c
#	gcc -O2 -Wall -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/numpy -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -I/System/Library/Frameworks/Python.framework/Versions/2.7/include -lpython2.7 -o clogistic.so clogistic.c

clean:
	rm -f clogistic.so clogistic.c
