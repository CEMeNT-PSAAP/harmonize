#pragma once

#include "basic.h"


namespace host {


void check_error(){

	adapt::gpurtError_t status = adapt::gpurtGetLastError();

	throw_on_error("",status);

}




void auto_throw(adapt::gpurtError_t value){
	if ( value != adapt::gpurtSuccess ) { throw value; }
}


template<typename T>
T* hardMalloc(size_t size){
	T* result;
	auto_throw( adapt::gpurtMalloc(&result, sizeof(T)*size) );
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
				adapt::gpurtError_t free_err = adapt::gpurtFree(adr);
				if(free_err != adapt::gpurtSuccess) {
					std::cerr << "ERROR: Failed to free device buffer managed by a DevBuf!\n";
					//throw std::runtime_error("Failed to free device buffer.");
				}
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	operator T*&() { return inner->adr; }

	__host__ void resize(size_t s){
		T* new_adr = hardMalloc<T>(s);
		size_t copy_count = ( s < inner->size ) ? s : inner->size;
		auto_throw( adapt::gpurtMemcpy(
			new_adr,
			inner->adr,
			sizeof(T)*copy_count,
			adapt::gpurtMemcpyDeviceToDevice
		) );
		if( inner->adr != NULL ) {
			auto_throw( adapt::gpurtFree(inner->adr) );
		}
		inner->size = s;
		inner->adr = new_adr;
	}

	__host__ void operator<<(std::vector<T> &other) {
		if( other.size() != inner->size ){
			resize(other.size());
		}
		auto_throw( adapt::gpurtMemcpy(
			inner->adr,
			other.data(),
			sizeof(T)*inner->size,
			adapt::gpurtMemcpyHostToDevice
		) );
	}

	__host__ void operator>>(std::vector<T> &other) {
		if( other.size() != inner->size ){
			other.resize(inner->size);
		}
		auto_throw( adapt::gpurtMemcpy(
			other.data(),
			inner->adr,
			sizeof(T)*inner->size,
			adapt::gpurtMemcpyDeviceToHost
		) );
	}


	__host__ void operator<<(T &other) {
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::gpurtMemcpy(
			inner->adr,
			&other,
			sizeof(T),
			adapt::gpurtMemcpyHostToDevice
		) );
	}

	__host__ void operator<<(T &&other) {
		T host_copy = other;
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( adapt::gpurtMemcpy(
			inner->adr,
			&host_copy,
			sizeof(T),
			adapt::gpurtMemcpyHostToDevice
		) );
	}

	
	__host__ void operator>>(T &other) {
		auto_throw( adapt::gpurtMemcpy(
			&other,
			inner->adr,
			sizeof(T),
			adapt::gpurtMemcpyDeviceToHost
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
			auto_throw( adapt::gpurtMemcpy(
				adr,
				&host_copy,
				sizeof(T),
				adapt::gpurtMemcpyHostToDevice
			) );
		}

		void pull_data(){
			//printf("Pulling data from %p\n",adr);
			auto_throw( adapt::gpurtMemcpy(
				&host_copy,
				adr,
				sizeof(T),
				adapt::gpurtMemcpyDefault
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
				adapt::gpurtError_t free_err = adapt::gpurtFree(adr);
				if(free_err != adapt::gpurtSuccess) {
					std::cerr << "ERROR: Failed to free device buffer managed by a DevObj!\n";
				}
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







