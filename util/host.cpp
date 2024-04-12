

namespace util {

namespace host {


bool check_error(){

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess){
		const char* err_str = cudaGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

	return (status != cudaSuccess);

}




void auto_throw(cudaError_t status){
	if ( status != cudaSuccess ) {
		std::string message = "GPU Runtime Error: ";
		message += cudaGetErrorString(status);
		throw std::runtime_error(message);
	}
}


}

}
