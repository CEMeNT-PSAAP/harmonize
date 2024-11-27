
test:
	@if which nvcc 2>/dev/null; \
	then \
		nvcc -x cu main.cpp test_launch.cpp -arch=sm_75 -o test.exe -g -lpthread ; \
	else \
		hipcc -x hip test_launch.cpp main.cpp -o test.exe -g -lpthread ; \
	fi

	#valgrind -s ./test.exe
	./test.exe



