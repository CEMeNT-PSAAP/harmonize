
test:
	@if which nvcc 2>/dev/null; \
	then \
		nvcc -x cu test.cpp -arch=sm_75 -o test.exe -g; \
	else \
		hipcc -x hip test.cpp -o test.exe -g; \
	fi

	#valgrind -s ./test.exe
	./test.exe



