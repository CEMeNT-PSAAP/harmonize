


#ifndef HARMONIZE_ASYNC_PROMISE
#define HARMONIZE_ASYNC_PROMISE

#include "../preamble/mod.h"


//! The type used as a discriminant (identifying value) between operation
//! types.
using OpDisc = unsigned int;

//! The `OpUnion` template struct serves to represent a set of operations as a type
//! so that it may be passed as a template parameter.
template<typename... TYPES>
struct OpUnion {};

//! The `Promise` template struct represents all information associated with an async
//! call to its opertion type, including all passed arguments as well as return
//! addresses.
template<typename OPERATION>
struct Promise;

//! The `PromiseUnion` template struct represents a union across a set of different
//! promise types. Through this template, a field can generically store data
//! corresponding to a variety of different async calls.
template <typename OP_UNION>
union PromiseUnion;

//! The `PromiseEnum` template struct is simply the combination of a `PromiseUnion` and
//! a discriminant that annotates the contained type.
template <typename OP_UNION>
struct PromiseEnum;

//! The `VoidState` struct is an empty struct, used as a default state type, for states
//! that are not defined the supplied program specification.
struct VoidState {};

//! The `ReturnOp` operation type is an internal type used to represent the resolution
//! of dependencies and the fullfilment of futures.
struct ReturnOp;

//! The `RamappingBarrier` template struct is a barrier that can store an arbitrary number
//! of promises with any operation type contained by its operation set, automatically
//! coalescing promises of equivalent type into work links.
template<typename OP_SET, typename ADR_TYPE = unsigned int>
struct RemappingBarrier;

//! The `UnitBarrier` template struct is a barrier that can hold up to one promise with
//! an operation type contained by its operation set.
template<typename OP_SET, typename ADR_TYPE = unsigned int>
struct UnitBarrier;

//! The `Future` template struct ties a value field of a certain type to a barrier, allowing
//! operations to await the definition of the value.
template<typename TYPE, typename BARRIER>
struct Future;



//! The `OpUnionLookup` template struct is used to derive type information about
//! `OpUnion` specializations, particularly whether or not a given type is contained
//! within the union.
template <typename QUERY, typename OP_UNION>
struct OpUnionLookup;

//! The base case of the `OpUnionLookup` template struct. This terminates all recursively
//! driven definitions from non-base cases.
template <typename QUERY>
struct OpUnionLookup <QUERY,OpUnion<>>
{
	static const bool   CONTAINED = false;
	static const OpDisc DEPTH     = 0;
	static const OpDisc Q_LEVEL   = DEPTH;
	static const OpDisc DISC      = DEPTH;
};

//! The recursive case of the `OpUnionLookup` template struct. This struct reports information
//! such as whether or not a type is contained within the supplied `OpUnion` specialization,
//! how many operations are stored in the `OpUnion`, and what the discriminant value of a
//! given operation type is for the union.
template <typename QUERY, typename HEAD, typename... TAIL>
struct OpUnionLookup <QUERY, OpUnion<HEAD, TAIL...> >
{
	static const bool   MATCHES   = std::is_same<QUERY,HEAD>::value ;
	static const bool   CONTAINED = MATCHES || (OpUnionLookup<QUERY,OpUnion<TAIL...>>::CONTAINED);
	static const OpDisc DEPTH     = OpUnionLookup<QUERY,OpUnion<TAIL...>>::DEPTH + 1;
	static const OpDisc Q_LEVEL   = MATCHES ? DEPTH : (OpUnionLookup<QUERY,OpUnion<TAIL...>>::Q_LEVEL);
	static const OpDisc DISC      = DEPTH - Q_LEVEL;
};




//! The `OpUnionAppend` template struct simply defines an internal type which is the
//! concatenation of the first type parameter onto the second `OpUnion` specialization
//! type parameter.
template<typename HEAD, typename TAIL_UNION>
struct OpUnionAppend;

template<typename HEAD, typename... TAIL>
struct OpUnionAppend<HEAD,OpUnion<TAIL...>>
{
	using Type = OpUnion<HEAD,TAIL...>;
};


//! The `OpUnionPair` template is used to determine information about pairs of
//! `OpUnion` specializations at compile time.
template <typename LEFT, typename RIGHT>
struct OpUnionPair;

template <>
struct OpUnionPair<OpUnion<>,OpUnion<>>
{
	static const bool LEFT_SUBSET  = true;
	static const bool RIGHT_SUBSET = true;
	static const bool EQUAL        = true;
};

template <typename LEFT_HEAD, typename... LEFT_TAIL>
struct OpUnionPair<OpUnion<LEFT_HEAD,LEFT_TAIL...>,OpUnion<>>
{
	static const bool LEFT_SUBSET  = false;
	static const bool RIGHT_SUBSET = true;
	static const bool EQUAL        = false;
};

template <typename RIGHT_HEAD, typename... RIGHT_TAIL>
struct OpUnionPair<OpUnion<>,OpUnion<RIGHT_HEAD,RIGHT_TAIL...>>
{
	static const bool LEFT_SUBSET  = true;
	static const bool RIGHT_SUBSET = false;
	static const bool EQUAL        = false;
};

template <typename LEFT_HEAD, typename... LEFT_TAIL, typename RIGHT_HEAD, typename... RIGHT_TAIL>
struct OpUnionPair<OpUnion<LEFT_HEAD,LEFT_TAIL...>,OpUnion<RIGHT_HEAD,RIGHT_TAIL...>>
{
	using Left       = OpUnion<LEFT_HEAD,LEFT_TAIL...>;
	using LeftTail   = OpUnion<LEFT_TAIL...>;
	using Right      = OpUnion<RIGHT_HEAD,RIGHT_TAIL...>;
	using RightTail  = OpUnion<RIGHT_TAIL...>;

	static const bool LEFT_SUBSET  = OpUnionLookup<LEFT_HEAD,Right>::CONTAINED && OpUnionPair<LeftTail,Right>::LEFT_SUBSET;
	static const bool RIGHT_SUBSET = OpUnionLookup<RIGHT_HEAD,Left>::CONTAINED && OpUnionPair<RightTail,Left>::LEFT_SUBET;
	static const bool EQUAL        = LEFT_SUBSET && RIGHT_SUBSET;
};



//! The `OpReturnFilter` tempalate struct is used to find the subset of a given
//! operation union that has a given return type. This is useful for checking
//! the validity of promises being used as return values for an operation.
template<typename RETURN, typename OP_UNION>
struct OpReturnFilter;

template<typename RETURN>
struct OpReturnFilter<RETURN,OpUnion<>>
{
	using Type = OpUnion<>;
};

template<typename RETURN, typename HEAD, typename... TAIL>
struct OpReturnFilter<RETURN,OpUnion<HEAD,TAIL...>>
{

	template<typename H,typename... T>
	static typename std::enable_if<
		! ( std::is_same< typename Promise<H>::Return,RETURN>::value ) ,
		typename OpReturnFilter<RETURN,OpUnion<T...>>::Type
	>::type
	filtered ();

	template<typename H,typename... T>
	static typename std::enable_if<
		std::is_same< typename Promise<H>::Return,RETURN>::value,
		typename OpUnionAppend<H,typename OpReturnFilter<RETURN,OpUnion<T...>>::Type>::Type
	>::type
	filtered ();

	using Type = decltype(filtered<HEAD,TAIL...>());
};



//! A `TaggedSemaphore` is a semaphore that is tagged with information
//! identifying the type that contains the tag. This is useful for determining
//! the type of a barrier/future that contains the semaphore.
struct TaggedSemaphore {
	unsigned int sem;
	unsigned int tag;

	TaggedSemaphore() = default;
	__host__ __device__ TaggedSemaphore(unsigned int sem_val, unsigned int tag_val)
		: sem(sem_val)
		, tag(tag_val)
	{}
};



//! The `LazyLoad` template struct wraps a reference to a value of a type mathcing
//! its type parameter. Instances of this template struct represent a parameter
//! to an asynchronous call that should be filled in by an asynchronous load from
//! a location in global memory.
template <typename TYPE>
struct LazyLoad {
	using Type = TYPE;
	Type& reference;

	__host__ __device__ LazyLoad<TYPE>(Type& ref) : reference(ref) {}

	__host__ __device__ operator TYPE ()
	{
		return reference;
	}
};




//! The `has_load` template function returns true if and only if the supplied
//! parameters contains at least one argument that specializes the `LazyLoad`
//! template struct.
template <typename... TYPES>
__host__ __device__ bool has_load(TYPES...){
	return false;
}

template <typename HEAD, typename... TAIL>
__host__ __device__ bool has_load(HEAD head, TAIL... tail){
	return has_load(tail...);
}

template <typename HEAD, typename... TAIL>
__host__ __device__ bool has_load(LazyLoad<HEAD> head, TAIL... tail){
	return true;
}



//! The `ArgTuple` template struct represents the storage of an arbitrary
//! number of fields of arbitrary type, and is used to store the argument
//! data of asynchonous calls inside promises.
template <typename... TAIL>
struct ArgTuple;

template <>
struct ArgTuple <>
{
	__host__ __device__ ArgTuple<>(){}

	#ifdef ASYNC_LOADS
	template<typename PROGRAM>
	__device__ void async_load(PROGRAM prog) {}
	#endif
};

template <typename HEAD, typename... TAIL>
struct ArgTuple <HEAD, TAIL...>
{
	using Tail = ArgTuple<TAIL...>;

	HEAD head;
	Tail tail;

	__host__ __device__ ArgTuple<HEAD,TAIL...> (HEAD h, TAIL... t)
		: head(h)
		, tail(t...)
	{}

	__host__ __device__ ArgTuple<HEAD,TAIL...> (LazyLoad<HEAD> h, TAIL... t)
		: head(h)
		, tail(t...)
	{}

	template<typename PROGRAM, typename LOAD_HEAD, typename... LOAD_TAIL>
	__device__ void async_load(PROGRAM prog, LOAD_HEAD, LOAD_TAIL...);


	#ifdef ASYNC_LOADS
	//! If asynchronous loading is enabled, an `ArgTuple` can be filled in with a
	//! mixture of direct value write and asynchronous loads. This is done through
	//! the async_load funtion, which direcctly writes all arguments passed in by
	//! value and initiates async loads for all arguments passed in as `LazyLoad`
	//! structs.
	template<typename PROGRAM, typename... LOAD_TAIL>
	__device__ void async_load(PROGRAM prog, HEAD h, LOAD_TAIL... t)
	{
		head = h;
		tail.async_load(prog,t...);
	}

	template<typename PROGRAM, typename... LOAD_TAIL>
	__device__ void async_load(PROGRAM prog, LazyLoad<HEAD> h, LOAD_TAIL... t)
	{
		cuda::memcpy_async(&head,&h.reference,sizeof(HEAD),prog._grp_ctx.load_barrier);
		tail.async_load(prog,t...);
	}
	#endif

};



//! The `OpType` template struct is used to deduce the arguments and return
//! type of the supplied template argument. This is used to determine information
//! about the function signatures of operations.
template<typename TYPE>
struct OpType;

template<typename RETURN, typename... ARGS>
struct OpType < RETURN (*) (ARGS...) > {
	typedef RETURN Return;
	typedef ArgTuple<ARGS...> Args;
};





//! The `ReturnAdr` template struct is used to represent the destination
//! semaphore and (if applicable) future values that must be decremented
//! or filled in to resolve an operation. For void return types, only
//! a semaphore address is stored. In cases where there is no destination
//! value field, the value address is set to NULL. Likewise, promises
//! that have no barrier/future to resolve into are given a NULL semaphore
//! address.
template<typename TYPE>
struct ReturnAdr {
	using Type = TYPE;

	TaggedSemaphore  *base;
	Type             *dest;
};

template<>
struct ReturnAdr<void> {
	using Type = void;

	TaggedSemaphore  *base;
};



//! The `Return` template struct is used to represent the set of all types
//! that can be returned from an operation's `eval` function. This includes
//! returning by value, returning with a promise that returns the appropriate
//! type, or returning a future that will eventually define a value of the
//! appropriate type.
template <typename PROGRAM, typename TYPE>
struct Return {

	using Type        = TYPE;
	using ProgramType = PROGRAM;
	using ValidSet    = typename OpReturnFilter<TYPE,typename ProgramType::OpSet>::Type;
	using OpSet       = typename ProgramType::OpSet;
	using EnumType    = PromiseEnum<OpSet>;
	using RetAdrType  = ReturnAdr<TYPE>;
	using RFutureType = Future<Type,RemappingBarrier<typename PROGRAM::OpSet>>;
	using UFutureType = Future<Type,UnitBarrier     <typename PROGRAM::OpSet>>;

	enum Form { VALUE, PROMISE, FUTURE };

	union Data {
		Type             value;
		EnumType         promise;
		RetAdrType       future;

		Data(Type        val) : value   (val) {}
		Data(EnumType    prm) : promise (prm) {}
		Data(RetAdrType  fut) : future  (fut) {}
	};

	Form form;
	Data data;

	//! The `return_guard` function causes a compile time error when promises that do
	//! not belong in the operation set of a `Return` are passed in as initializing
	//! arguments. This helps to replace the more cryptic errors that come about when
	//! trying to pass in an inappropriate `Promise` type.
	template<typename OP_TYPE>
	__host__ __device__ static constexpr Promise<OP_TYPE> return_guard(Promise<OP_TYPE> promise)
	{
		static_assert(
			PromiseUnion<ValidSet>::template Lookup<OP_TYPE>::type::CONTAINED,
			"\n\nTYPE ERROR: Type of returned promise does not match return type of "
			"operation.\n\n"
			"SUGGESTION: Ensure that the operation of the returned promise returns the "
			"same type as the returning operation.\n\n"
		);
		return promise;
	}

	template<typename OP_TYPE>
	__host__ __device__ Return<ProgramType,Type>(Promise<OP_TYPE> promise)
		: form(Form::PROMISE)
		, data(return_guard(promise))
	{}

	__host__ __device__ Return<ProgramType,Type>(Type value)
		: form(Form::VALUE)
		, data(value)
	{}

	__host__ __device__ Return<ProgramType,Type>(RetAdrType future)
		: form(Form::FUTURE)
		, data(future)
	{}


/*
	__host__ __device__ void resolve(PROGRAM program, RetAdrType ret){
		switch(form){
			case Form::VALUE:
				ret.resolve(data.value);
				break;
			case Form::PROMISE:
				data.promise.ret = ret;
				program.async_call(data.promise);
				break;
			case Form::Future:

				data.future.await(Promise<ReturnOp>(ret.base,ret.data,));
				break;
			default: //impossible
		}
	}
*/

};


//! The `Promise` template struct represents the arguments and return information associated
//! with an asynchronous call to the input operation type parameter.
template<typename OPERATION>
struct Promise
{

	typedef typename OpType< typename OPERATION::Type >::Return Return;
	typedef typename OpType< typename OPERATION::Type >::Args   Args;

	ReturnAdr<Return> ret;
	Args args;

	//! The `inner_eval` template function is responsible for unpacking the components of an
	//! `arg_tuple` so that they may be passed to an operation's `eval` function as individual
	//! arguments.
	template<typename PROGRAM, typename... TUPLE_ARGS, typename... UNROLL_ARGS>
	__device__ Return inner_eval(PROGRAM program, ArgTuple<TUPLE_ARGS...> tuple_args, UNROLL_ARGS... unroll_args)
	{
		return inner_eval(program,tuple_args.tail,unroll_args...,tuple_args.head);
	}


	template<typename PROGRAM, typename... UNROLL_ARGS>
	__device__ Return inner_eval(PROGRAM program, ArgTuple<> empty_tuple_args, UNROLL_ARGS... unroll_args)
	{
		return OPERATION::template eval<PROGRAM>(program,unroll_args...);
	}


	//! The call operator accepts the executing program as its only input and passes its
	//! argument values alongside the program into the operation's `eval` function.
	template<typename PROGRAM>
	__device__ Return operator() (PROGRAM program) {
		return inner_eval(program,args);
	}

	//! Instances of any given `Promise` specialization are constructed by passing in the
	//! values of all arguments in the corresponding operation's `eval` function signature.
	template<typename... ARGS>
	__host__ __device__ Promise<OPERATION> ( ARGS... a ) : args(a...) {}


	#ifdef ASYNC_LOADS
	//! If asynchronous loading is enabled, a `Promise` can be filled in (after default
	//! initialization) with a pack of argument values intermixed with appropriate
	//! `LazyLoad` values. This fills in all arguments passed in by value and initiates
	//! the asynchronous loading of all arguments replaced with `LazyLoad` structs.
	template<typename PROGRAM, typename... ARGS>
	__device__ void async_load ( PROGRAM prog, ARGS... a ) {
		args.async_load(prog, a...);
	}
	#endif

};








//!
//! The base case of the `PromiseUnion` template union defines empty functions to cap off the
//! recursion of non-base cases when evaluating promises.
//!
template <>
union PromiseUnion <OpUnion<>> {

	struct Info {
		static const OpDisc COUNT = 0;
	};

	template <typename PROGRAM, typename TYPE>
	__host__  __device__ void rigid_eval( PROGRAM program ) {
		return;
	}

	template <typename PROGRAM>
	__host__  __device__ void loose_eval( PROGRAM program, OpDisc op_disc ) {
		return;
	}

	__host__ __device__ void dyn_copy_as( OpDisc op_disc, PromiseUnion<OpUnion<>>& other){ }

	PromiseUnion<OpUnion<>> () = default;

	#ifdef ASYNC_LOADS
	template<typename PROGRAM>
	__device__ void async_load ( PROGRAM prog ) {}
	#endif
};



//!
//! The recursive case of the `PromiseUnion` template union defines the `cast()`, `rigid_eval()`,
//! and `loose_eval()` template functions for the async function/parameter type corresponding to
//! the first template argument.
//!
template <typename HEAD, typename... TAIL>
union PromiseUnion<OpUnion<HEAD, TAIL...>>
{

	using Head      = Promise<HEAD>;
	using Tail      = PromiseUnion<OpUnion<TAIL...>>;

	using OpSet     = OpUnion<HEAD,TAIL...>;
	using TailOpSet = OpUnion<TAIL...>;

	Head head_form;
	Tail tail_form;

	public:

	//! For some compilers, a union cannot diretly contain static const values.
	//! To get around this, we wrap our compile time type information for PromiseUnions
	//! with internl types such as the `Info` struct type or the `Looup` template
	//! struct.
	struct Info {
		static const OpDisc COUNT = sizeof...(TAIL) + 1;
		static const OpDisc INDEX = OpUnionLookup<HEAD,OpSet>::DISC;
	};

	template <typename TYPE>
	struct Lookup { typedef OpUnionLookup<TYPE,OpSet> type; };


	//! The `cast` template function allows a `PromiseUnion` to be cast to any `Promise`
	//! specialization with a corresponding operation type contained by the union's
	//! operation set.
	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		std::is_same<TYPE,HEAD>::value,
		Promise<TYPE>&
	>::type
	cast() {
		return head_form;
	}

	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && OpUnionLookup<TYPE,TailOpSet>::CONTAINED,
		Promise<TYPE>&
	>::type
	cast(){
		return tail_form.template cast<TYPE>();
	}

	//! This specialization of the `cast` template function provide a useful compiler
	//! error for cases when the supplied operation type is not contained by the operation
	//! set of the `PromiseUnion`
	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && (!OpUnionLookup<TYPE,TailOpSet>::CONTAINED),
		Promise<TYPE>&
	>::type
	cast (){
		static_assert(
			(!OpUnionLookup<TYPE,TailOpSet>::CONTAINED),
			"\n\nTYPE ERROR: Promise type does not exist in union.\n\n"
		);
	}


	//! The `rigid_eval` template function evaluates the contained union as a `Promise`
	//! corresponding to the supplied operation type. This allows the compiler to deduce
	//! which type of operation is being performed and optimize out branches it knows
	//! will never be taken.
	template <typename PROGRAM, typename TYPE >
	__host__  __device__ typename std::enable_if<
		std::is_same<TYPE,HEAD>::value,
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		head_form(program);
	}

	template <typename PROGRAM, typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && OpUnionLookup<TYPE,TailOpSet>::CONTAINED,
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		tail_form.template rigid_eval<PROGRAM,TYPE>(program);
	}

	template <typename PROGRAM, typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && (!OpUnionLookup<TYPE,TailOpSet>::CONTAINED),
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		static_assert(
			(!OpUnionLookup<TYPE,TailOpSet>::CONTAINED),
			"\n\nTYPE ERROR: Promise type does not exist in union.\n\n"
		);
	}


	//! The `loose_eval` template function evaluates the contained union as a `Promise`
	//! corresponding to an operation with the supplied discriminant value. This should
	//! be used only in situations where the type of the operation is not known at
	//! compile time.
	template <typename PROGRAM>
	__host__  __device__ void loose_eval (
		PROGRAM program,
		OpDisc disc
	) {
		if(disc == Info::INDEX){
			head_form(program);
		} else {
			tail_form. template loose_eval<PROGRAM>(program,disc-1);
		}

	}


	//! The copy_as and dyn_copy_as functions copy data from one union to another
	//! as a particular type of `Promise`.
	template <typename TYPE>
	__host__ __device__ void copy_as(PromiseUnion<OpSet>& other){
		cast<TYPE>() = other.template cast<TYPE>();
	}


	__host__ __device__ void dyn_copy_as(OpDisc disc, PromiseUnion<OpSet>& other){
		if( disc == Info::INDEX ){
			cast<HEAD>() = other.cast<HEAD>();
		} else {
			tail_form.dyn_copy_as(disc-1,other.tail_form);
		}
	}

	__host__ __device__ PromiseUnion<OpSet> () : tail_form() {}

	//! A `PromiseUnion` can be intialized by a `Promise` of any type contained by
	//! its operation set. All other `Promise` types cause a compile time error to
	//! be thrown alongside the helpful message below.
	template< typename OP_TYPE>
	__host__ __device__ PromiseUnion<OpSet> ( Promise<OP_TYPE> prom ) {
		static_assert(
			OpUnionLookup<OP_TYPE,OpSet>::CONTAINED,
			"\n\nTYPE ERROR: Type of assigned promise does not exist in promise union.\n\n"
			"SUGGESTION: Double-check the signature of the promise union and make sure "
			"its OpUnion template parameter contains the desired operation type.\n\n"
			"SUGGESTION: Double-check the type signature of the promise and make sure "
			"it is the correct operation type.\n\n"
		);
		cast<OP_TYPE>() = prom;
	}


};





//! A `PromiseEnum` is just a tagged version of a `PromiseUnion`, combining the
//! union type with a discriminant that annotates the type of `Promise` contained
//! by the union.
template <typename OP_SET>
struct PromiseEnum {

	using UnionType = PromiseUnion<OP_SET>;

	UnionType data;
	OpDisc    disc;

	PromiseEnum() = default;

	__host__ __device__ PromiseEnum<OP_SET>(UnionType uni, OpDisc d)
		: data(uni)
		, disc(d)
	{}


	template<typename OP_TYPE>
	__host__ __device__ constexpr static Promise<OP_TYPE> promise_guard (Promise<OP_TYPE> promise){
		static_assert(
			UnionType::template Lookup<OP_TYPE>::type::CONTAINED,
			"\n\nTYPE ERROR: Type of assigned promise does not exist in promise enum.\n\n"
			"SUGGESTION: Double-check the signature of the promise enum and make sure "
			"its OpUnion template parameter contains the desired operation type.\n\n"
			"SUGGESTION: Double-check the type signature of the promise and make sure "
			"it is the correct operation type.\n\n"
		);
		return promise;
	}


	template<typename OP_TYPE>
	__host__ __device__ PromiseEnum<OP_SET>(Promise<OP_TYPE> promise)
		: data(promise_guard(promise))
		, disc(UnionType::template Lookup<OP_TYPE>::type::DISC)
	{}


};



#endif

