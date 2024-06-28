
#ifndef HARMONIZE_UTIL_HOST
#define HARMONIZE_UTIL_HOST

namespace util {

namespace host {


static bool check_error(){

	adapt::rtError_t status = adapt::rtGetLastError();

	if(status != adapt::rtSuccess){
		const char* err_str = adapt::rtGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

	return (status != adapt::rtSuccess);

}




static void auto_throw(adapt::rtError_t status){
	if ( status != adapt::rtSuccess ) {
		std::string message = "GPU Runtime Error: ";
		message += adapt::rtGetErrorString(status);
		throw std::runtime_error(message);
	}
}


template<typename T>
T* hardMalloc(size_t size){
	T* result;
	auto_throw( adapt::rtMalloc(&result, sizeof(T)*size) );
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
				adapt::rtFree(adr);
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	operator T*&() { return inner->adr; }

	__host__ void resize(size_t s){
		T* new_adr = hardMalloc<T>(s);
		size_t copy_count = ( s < inner->size ) ? s : inner->size;
		auto_throw( adapt::rtMemcpy(
			new_adr,
			inner->adr,
			sizeof(T)*copy_count,
			adapt::rtMemcpyDeviceToDevice
		) );
		if( inner->adr != NULL ) {
			auto_throw( adapt::rtFree(inner->adr) );
		}
		inner->size = s;
		inner->adr = new_adr;
	}

	__host__ void operator<<(std::vector<T> &other) {
		if( other.size() != inner->size ){
			resize(other.size());
		}
		auto_throw( adapt::rtMemcpy(
			inner->adr,
			other.data(),
			sizeof(T)*inner->size,
			adapt::rtMemcpyHostToDevice
		) );
	}

	__host__ void operator>>(std::vector<T> &other) {
		if( other.size() != inner->size ){
			other.resize(inner->size);
		}
		auto_throw( adapt::rtMemcpy(
			other.data(),
			inner->adr,
			sizeof(T)*inner->size,
			adapt::rtMemcpyDeviceToHost
		) );
	}


	__host__ void operator<<(T &other) {
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::rtMemcpy(
			inner->adr,
			&other,
			sizeof(T),
			adapt::rtMemcpyHostToDevice
		) );
	}

	__host__ void operator<<(T &&other) {
		T host_copy = other;
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::rtMemcpy(
			inner->adr,
			&host_copy,
			sizeof(T),
			adapt::rtMemcpyHostToDevice
		) );
	}


	__host__ void operator>>(T &other) {
		auto_throw( adapt::rtMemcpy(
			&other,
			inner->adr,
			sizeof(T),
			adapt::rtMemcpyDeviceToHost
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
			auto_throw( adapt::rtMemcpy(
				adr,
				&host_copy,
				sizeof(T),
				adapt::rtMemcpyHostToDevice
			) );
		}

		void pull_data(){
			//printf("Pulling data from %p\n",adr);
			auto_throw( adapt::rtMemcpy(
				&host_copy,
				adr,
				sizeof(T),
				adapt::rtMemcpyDefault
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
				adapt::rtFree(adr);
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

