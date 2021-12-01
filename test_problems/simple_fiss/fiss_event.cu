

#include "fiss_common.cu"

using namespace util;


#define DYN

typedef ProgramStateDef<SimParams,VoidState,VoidState> ProgState;

enum class Fn { Neutron };

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);

typedef  EventProgram < PromiseUnion<Fn::Neutron>, ProgState > ProgType;


DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {


	if( arg == Adr<unsigned int>::null ){
		printf("{   Bad argument!   }");
		return;
	}


	Neutron n;

	#ifdef DYN
	n = (*global.neutron_pool)[arg];
	#else
	n =  global.neutron_buffer[arg];
	#endif

	int result = 0;
	for(int i=0; i < global.horizon; i++){
		result = step_neutron(global,n);
		if( result != 0 ){
			break;
		}
	}

	#ifdef DYN

	#ifdef FILO
	unsigned int last = n.next;
	#endif

	for(int i=0; i<result; i++){
		Neutron new_neutron(n);

		#ifdef FILO
		new_neutron.next = last;
		#endif

		unsigned int index = global.neutron_pool->alloc(_thread_context.rand_state);
		if( index != Adr<unsigned int>::null ){
			unsigned int old = atomicCAS(&(*global.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad fiss alloc %d at %d}\n",old,index);
			}

			#ifdef FILO
			last = index;
			#endif

			(*global.neutron_pool)[index] = new_neutron;

			#ifdef FILO
			if( i == (result-1) ){
				QUEUE_EVENT(Fn::Neutron,index);
			}
			#else
			QUEUE_EVENT(Fn::Neutron,index);
			#endif
		} else {
			printf("{Fiss alloc fail}");
		}


	}



	if( result == 0 ) {
		(*global.neutron_pool)[arg] = n;
		QUEUE_EVENT(Fn::Neutron,arg);		
	}
	else {

		#ifdef FILO
		if( (result < 0) && (n.next != Adr<unsigned int>::null) ){
			QUEUE_EVENT(Fn::Neutron,n.next);
		}
		#endif
		unsigned int old = atomicCAS(&(*global.neutron_pool)[arg].checkout,1u,0u);
		if( old != 1 ){
			printf("{Bad dealloc %d at %d}",old,arg);
		}
		global.neutron_pool->free(arg,_thread_context.rand_state);
	}

	#else
	global.neutron_buffer[arg] = n;
	QUEUE_EVENT(Fn::Neutron,arg);		
	#endif

}


DEF_INITIALIZE(ProgType) {


}


DEF_FINALIZE(ProgType) {


}


DEF_MAKE_EVENTS(ProgType) {


	unsigned int id;

	if( QUEUE_FILL_FRACTION(Fn::Neutron) > 0.01 ){
		/*
		if( threadIdx.x == 0 ){
			printf("{Early escape.}");
		}
		// */
		return false;
	}

	Iter<unsigned int> iter = global.source_id_iter->leap(8u);//(14u);

	while(iter.step(id)){
		Neutron n(id,0.0,0.0,0.0,0.0);

		#ifdef FILO
		n.next = Adr<unsigned int>::null;
		#endif

		#ifdef DYN
		unsigned int index = global.neutron_pool->alloc(_thread_context.rand_state);
		while(index == Adr<unsigned int>::null){
			printf("{Unable to alloc at %d}",index);
			index = global.neutron_pool->alloc(_thread_context.rand_state);
		}
		
		if( (index != Adr<unsigned int>::null) ) { //&& (index != 0) ){
			(*global.neutron_pool)[index] = n;
			unsigned int old = atomicCAS(&(*global.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad alloc %d at %d}\n",old,index);
			}
			PROCESS_EVENT(Fn::Neutron,index);
		}
		#else
		global.neutron_buffer [id] = n;
		PROCESS_EVENT(Fn::Neutron,id);
		#endif
		
	}

	return !global.source_id_iter->done();

}



int main(int argc, char *argv[]){


	ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	CommonContext com(args);

	cudaDeviceSynchronize();
		
	check_error();
	
	ProgType::Instance instance = ProgType::Instance(0x100000,com.params);
	cudaDeviceSynchronize();
	check_error();

	do {
		exec<ProgType>(instance,wg_count,24);
		cudaDeviceSynchronize();
		check_error();
	} while ( ! instance.complete() );

	
	return 0;

}

