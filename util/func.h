




#if 0


template<typename T>
class Option {

	T    data;
	bool some;

	operator bool() { return some; }

	T unwrap()
	{
		if ( some ) {
			return data;
		}
		throw std::string("Attempted to unwrap empty Option.");
	}

};



template<typename OKAY, typename FAIL>
class Result {

	union OkayOrFail {
		OKAY okay;
		FAIL fail;
	};

	OkayOrFail data;
	bool       okay;


	operator bool() { return okay; }

	Result<OKAY,FAIL>(OKAY value){
		data.okay = value;
		okay = true;
	}

	Result<OKAY,FAIL>(FAIL value){
		data.fail = value;
		okay = false;
	}

	static Result<OKAY,FAIL> wrap(OKAY value){
		Result<OKAY,FAIL> result;
		result.data.okay = value;
		result.okay = true;
		return result;
	}
	
	static Result<OKAY,FAIL> wrap_fail(FAIL value){
		Result<OKAY,FAIL> result;
		result.data.fail = value;
		result.okay = false;
		return result;
	}

	OKAY unwrap()
	{
		if ( okay ) {
			return data.okay;
		}
		throw std::string("Attempted to unwrap failed Result.");
	}

	FAIL unwrap_fail()
	{
		if ( !okay ) {
			return data.fail;
		}
		throw std::string("Attempted to unwrap failed Result.");
	}


	OKAY unwrap_or(OKAY other){
		return okay ? data.okay : other;
	}
	
	FAIL unwrap_fail_or(FAIL other){
		return !okay ? data.fail : other;
	}

	
	template<typename FUNCTION>
	auto and_then(FUNCTION f) -> Result<decltype(f(data.okay)),FAIL>
	{
		typedef Result<decltype(f(data.okay)),FAIL> RetType;

		if( ! okay ){
			return RetType::wrap_fail(data.fail);
		} else {
			return RetType::wrap_okay(f(data.okay));
		}
	}



};




#endif






