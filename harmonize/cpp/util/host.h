
#ifndef HARMONIZE_UTIL_HOST
#define HARMONIZE_UTIL_HOST

namespace util {

namespace host {


static bool check_error(){

	adapt::GPUrtError_t status = adapt::GPUrtGetLastError();

	if(status != adapt::GPUrtSuccess){
		const char* err_str = adapt::GPUrtGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

	return (status != adapt::GPUrtSuccess);

}




static void auto_throw(adapt::GPUrtError_t status){
	if ( status != adapt::GPUrtSuccess ) {
		std::string message = "GPU Runtime Error: ";
		message += adapt::GPUrtGetErrorString(status);
		throw std::runtime_error(message);
	}
}


template<typename T>
T* hardMalloc(size_t size){
	T* result;
	auto_throw( adapt::GPUrtMalloc(&result, sizeof(T)*size) );
	return result;
}



template<typename T>
class DevBuf {

	protected:

	struct Inner {

		size_t  size;
		T       *adr;

		Inner(T *adr_val, size_t size_val)
		: adr (adr_val )
		, size(size_val)
		{}

		~Inner() {
			if ( adr != NULL) {
				auto _ = adapt::GPUrtFree(adr);
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	operator T*&() { return inner->adr; }

	__host__ void resize(size_t s){
		T* new_adr = hardMalloc<T>(s);
		size_t copy_count = ( s < inner->size ) ? s : inner->size;
		auto_throw( adapt::GPUrtMemcpy(
			new_adr,
			inner->adr,
			sizeof(T)*copy_count,
			adapt::GPUrtMemcpyDeviceToDevice
		) );
		if( inner->adr != NULL ) {
			auto_throw( adapt::GPUrtFree(inner->adr) );
		}
		inner->size = s;
		inner->adr = new_adr;
	}

	__host__ void operator<<(std::vector<T> &other) {
		if( other.size() != inner->size ){
			resize(other.size());
		}
		auto_throw( adapt::GPUrtMemcpy(
			inner->adr,
			other.data(),
			sizeof(T)*inner->size,
			adapt::GPUrtMemcpyHostToDevice
		) );
	}

	__host__ void operator>>(std::vector<T> &other) {
		if( other.size() != inner->size ){
			other.resize(inner->size);
		}
		auto_throw( adapt::GPUrtMemcpy(
			other.data(),
			inner->adr,
			sizeof(T)*inner->size,
			adapt::GPUrtMemcpyDeviceToHost
		) );
	}


	__host__ void operator<<(T &other) {
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::GPUrtMemcpy(
			inner->adr,
			&other,
			sizeof(T),
			adapt::GPUrtMemcpyHostToDevice
		) );
	}

	__host__ void operator<<(T &&other) {
		T host_copy = other;
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::GPUrtMemcpy(
			inner->adr,
			&host_copy,
			sizeof(T),
			adapt::GPUrtMemcpyHostToDevice
		) );
	}


	__host__ void operator>>(T &other) {
		auto_throw( adapt::GPUrtMemcpy(
			&other,
			inner->adr,
			sizeof(T),
			adapt::GPUrtMemcpyDeviceToHost
		) );
	}


	DevBuf<T> (T* adr, size_t size)
		: inner(new Inner(adr,size))
	{}

	DevBuf<T> ()
		: DevBuf<T>((T*)NULL,(size_t)0)
	{}

	DevBuf<T> (size_t size)
		: DevBuf<T> (hardMalloc<T>(size),size)
	{}

	DevBuf<T> (T& value)
		: DevBuf<T>()
	{
		(*this) << value;
	}

	DevBuf<T> (T&& value)
		: DevBuf<T>()
	{
		(*this) << value;
	}


	template<typename... ARGS>
	static DevBuf<T> make(ARGS... args)
	{
		return DevBuf<T>( T(args...) );
	}


};




template<typename T>
class DevObj {

	protected:

	struct Inner {

		T *adr;
		T host_copy;


		void push_data(){
			//printf("Pushing data into %p\n",adr);
			auto_throw( adapt::GPUrtMemcpy(
				adr,
				&host_copy,
				sizeof(T),
				adapt::GPUrtMemcpyHostToDevice
			) );
		}

		void pull_data(){
			//printf("Pulling data from %p\n",adr);
			auto_throw( adapt::GPUrtMemcpy(
				&host_copy,
				adr,
				sizeof(T),
				adapt::GPUrtMemcpyDefault
			) );
		}

		template<typename... ARGS>
		Inner(T *adr_val, ARGS... args)
			: adr (adr_val)
			, host_copy(args...)
		{
			host_copy.host_init();
			push_data();
		}

		~Inner() {
			if ( adr != NULL) {
				//printf("Doing a free\n");
				pull_data();
				host_copy.host_free();
				util::host::auto_throw(adapt::GPUrtFree(adr));
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	void push_data(){
		inner->push_data();
	}

	void pull_data(){
		inner->pull_data();
	}

	operator T*()    { return inner->adr; }

	operator T()     { return inner->host_copy; }

	T& host_copy()   { return inner->host_copy; }

	template<typename... ARGS>
	DevObj<T>(ARGS... args)
		: inner(new Inner(hardMalloc<T>(1),args...))
	{}

};


}

}

#endif

