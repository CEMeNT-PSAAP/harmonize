
#ifndef HARMONIZE_UTIL_CLI
#define HARMONIZE_UTIL_CLI



namespace util {

namespace cli {



struct GraphStyle {

	int rows;
	int cols;
	const char** tiles;

	GraphStyle (unsigned int r, unsigned int c, const char** t)
		: rows(r)
		, cols(c)
		, tiles(t)
	{}

};


struct GraphShape {

	int width;
	int height;

	float x_min;
	float x_max;
	float y_min;
	float y_max;

};



static const char* BlockFillTiles[] = {
" ","▁","▂","▃","▄","▅","▅","▆","▇","█"
};
static const char* Block2x2FillTiles[] = {
	" ","▗","▐",
	"▖","▄","▟",
	"▌","▙","█"
};
static const char* Block2x3FillTiles[] = {
	" ","🬞","🬦","▐",
	"🬏","🬭","🬵","🬷",
	"🬓","🬱","🬹","🬻",
	"▌","🬲","🬺","█"
};
static const char* AsciiFillTiles[] = {
	" ",".",".","i","|",
	".","_","_","j","J",
	".","_","o","d","d",
	"i","L","b",":","4",
	"|","L","b","%","#"
};
static const char* BrailleFillTiles[25] = {
	"⠀","⢀","⢠","⢰","⢸",
	"⡀","⣀","⣠","⣰","⣸",
	"⡄","⣄","⣤","⣴","⣼",
	"⡆","⣆","⣦","⣶","⣾",
	"⡇","⣇","⣧","⣷","⣿"
};
static const GraphStyle BlockFill    = GraphStyle( 9, 1, BlockFillTiles  );
static const GraphStyle Block2x2Fill = GraphStyle( 2, 2, Block2x2FillTiles  );
static const GraphStyle Block2x3Fill = GraphStyle( 3, 2, Block2x3FillTiles  );
static const GraphStyle AsciiFill    = GraphStyle( 4, 2, AsciiFillTiles  );
static const GraphStyle BrailleFill  = GraphStyle( 4, 2, BrailleFillTiles);


static unsigned int lead_char_count(float val) {
	unsigned int result = 1;
	if( val >= 10.0 ){
		result = ceil(log10(val)+0.00000000001);
	} else if ( val < 0.0 ){
		if( val <= -10.0 ){
			result = ceil(log10(abs(val))) + 1;
		} else {
			result = 2;
		}
	}
	return result;
}



static void cli_graph(float* data, int size, GraphShape shape, GraphStyle style){

	// Local function for finding the width of a value's sign and leading digits
	// Find upper and lower bounds
	/*
	float max = data[0];
	float min = data[0];
	for( int i=0; i<size; i++){
		max = (data[i] > max) ? data[i] : max;
		min = (data[i] < min) ? data[i] : min;
	}
	*/

	// Determine how many columns will be alloted for leading digits and signs
	unsigned int min_lead_span = lead_char_count(shape.y_min);
	unsigned int max_lead_span = lead_char_count(shape.y_max);
	unsigned int lead_span = (min_lead_span > max_lead_span) ? min_lead_span : max_lead_span;

	// Iterate over the rows of the graph
	printf("%7.5f_\n",shape.y_max);
	for(int i=0; i<shape.height; i++){

		// Label tic marks of vertical axis
		float vert_span = shape.y_max-shape.y_min;
		float base = (shape.height-i-1)*vert_span/shape.height+shape.y_min;
		int whole_digits = lead_char_count(base);
		for(int i=whole_digits; i<lead_span; i++){
			printf(" ");
		}
		printf("%7.5f_",base);

		// Iterate over the elements of each row
		unsigned int x_iter = 0;
		float last = data[0];
		for(int j=0; j<shape.width; j++){
			unsigned int index = 0;
			// Iterate over each sub-column in each element
			for( int k=0; k<style.cols; k++){
				float val = 0.0;
				float count = 0.0;
				unsigned int limit = ((j*style.cols+k)*size)/(shape.width*style.cols);
				// Iterate over each value corresponding to each sub-column
				for(; x_iter < limit; x_iter++){
					val += data[x_iter];
					count += 1.0;
				}
				// Use last column value if no values correspond to current column
				if( count != 0.0 ) {
					val /= count;
					last = val;
				} else {
					val = last;
				}
				float rel_val = (val-base) / vert_span * shape.height * style.rows;
				unsigned int col_idx;
				col_idx = ( rel_val <= 0.0 )        ? 0          : rel_val;
				col_idx = ( rel_val >= style.rows ) ? style.rows : col_idx;
				index *= (style.rows + 1);
				index += col_idx;
			}
			printf("%s",style.tiles[index]);
		}


		printf("\n");
	}

	int   rule_size = 8*shape.width/2;
	char* rule_vals = new char[rule_size];
	memset(rule_vals,'\0',rule_size);

	printf("        ");
	for(int j=1; j<lead_span; j++){
		printf(" ");
	}
	for(int j=0; j<shape.width; j+=2){
		float tic_val = shape.x_min + ((shape.x_max-shape.x_min)/shape.width)*j;
		sprintf(&rule_vals[(8*j/2)],"%7.3f",tic_val);
		printf("\\ ");
	}
	printf("\n");
	for(int i=0; i<7; i++){
		printf("        ");
		for(int j=1; j<lead_span; j++){
			printf(" ");
		}
		for(int j=0; j<shape.width; j+=2){
			printf(" %c",rule_vals[(8*j/2)+i]);
		}
		printf("\n");
	}

	free(rule_vals);

}




static void cli_graph(float* data, int size, int width, int height, float low, float high){


	#if GRAPH_MODE == 4
	const int   cols = 5;
	const int   rows = 5;
	const char* lookup[25] = {
		"⠀","⡀","⡄","⡆","⡇",
		"⢀","⣀","⣄","⣆","⣇",
		"⢠","⣠","⣤","⣦","⣧",
		"⢰","⣰","⣴","⣶","⣷",
		"⢸","⣸","⣼","⣾","⣿"
	};
	#elif GRAPH_MODE == 1
	const char* lookup[9] = {
		" ","▖","▌",
		"▗","▄","▙",
		"▐","▟","█"
	};
	#else
	const char* lookup[25] = {
		" ",".",".","i","|",
		".","_","_","L","L",
		".","_","o","b","b",
		"i","j","d",":","%",
		"|","J","d","4","#"
	};
	#endif

	float max = 0;
	for( int i=0; i<size; i++){
		if( data[i] > max ){
			max = data[i];
		}
	}

	printf("Max is %f\n",max);

	int x_iter;
	float l_val, r_val;
	float last=0;

	int whole_digit_space = (max < 1) ? 1 : ceil(log10(max));

	printf("%7.5f_\n",max);
	for(int i=0; i<height; i++){
		float base = (height-i-1)*max/height;
		int whole_digits = (base < 1) ? 1 : ceil(log10(base));
		for(int i=whole_digits; i<whole_digit_space; i++){
			printf(" ");
		}
		printf("%7.5f_",base);
		x_iter = 0;
		for(int j=0; j<width; j++){
			l_val = 0;
			r_val = 0;
			int l_limit = (j*2*size)/(width*2);
			int r_limit = ((j*2+1)*size)/(width*2);
			float count = 0.0;
			for(; x_iter < l_limit; x_iter++){
				l_val += data[x_iter];
				//printf("%f,",data[x_iter]);
				count += 1.0;
			}
			l_val = ( count == 0.0 ) ? last : l_val / count;
			last = l_val;
			count = 0.0;
			for(; x_iter < r_limit; x_iter++){
				r_val += data[x_iter];
				count += 1.0;
			}
			r_val = ( count == 0.0 ) ? last : r_val / count;
			last = r_val;
			l_val = ( l_val - base )/max*height*4;
			r_val = ( r_val - base )/max*height*4;
			int l_idx = (l_val <= 0.0) ? 0 : ( (l_val >= 4.0) ? 4 : l_val );
			int r_idx = (r_val <= 0.0) ? 0 : ( (r_val >= 4.0) ? 4 : r_val );
			int str_idx = r_idx*5+l_idx;
			/*
			if( (str_idx < 0) || (str_idx >= 25) ){
				printf("BAD! [%d](%f:%d,%f:%d) -> (%d)",j,l_val,l_idx,r_val,r_idx,str_idx);
			}
			*/
			printf("%s",lookup[str_idx]);
		}
		printf("\n");
	}

	int   rule_size = 8*width/2;
	char* rule_vals = new char[rule_size];
	memset(rule_vals,'\0',rule_size);

	printf("        ");
	for(int j=1; j<whole_digit_space; j++){
		printf(" ");
	}
	for(int j=0; j<width; j+=2){
		float l_limit = low + ((high-low)/width)*j;
		sprintf(&rule_vals[(8*j/2)],"%7.3f",l_limit);
		printf("\\ ");
	}
	printf("\n");
	for(int i=0; i<7; i++){
		printf("        ");
		for(int j=1; j<whole_digit_space; j++){
			printf(" ");
		}
		for(int j=0; j<width; j+=2){
			printf(" %c",rule_vals[(8*j/2)+i]);
		}
		printf("\n");
	}

	free(rule_vals);

}






struct ArgSet
{

	int    argc;
	char** argv;

	int get_flag_idx(char* flag){
		for(int i=0; i<argc; i++){
			char* str = argv[i];
			if(    (str    != NULL )
				&& (str[0] == '-'  )
				&& (strcmp(str+1,flag) == 0)
						){
				return i;
			}
		}
		return -1;
	}


	char* get_flag_str(char* flag){
		int idx = get_flag_idx(flag);
		if( idx == -1 ) {
			return NULL;
		} else if (idx == (argc-1) ) {
			return (char*) "";
		}
		return argv[idx+1];
	}


	template<typename T>
	struct ArgVal {

		T     value;

		ArgVal(T val) : value(val)  {}

		ArgVal<T> operator| (T other) {
			return *this;
		}

		operator T() const {
			return value;
		}

	};


	struct ArgQuery {

		char* flag_str;
		char* value_str;


		template<typename T>
		bool scan_arg(char *str, T &dest) const {
			return false;
		}


		bool scan_arg(char *str, unsigned char &dest) const {
			return ( 0 < sscanf(str,"%hhu",&dest) );
		}

		bool scan_arg(char *str, unsigned short int &dest) const {
			return ( 0 < sscanf(str,"%hu",&dest) );
		}

		bool scan_arg(char *str, unsigned int &dest) const {
			return ( 0 < sscanf(str,"%u",&dest) );
		}

		bool scan_arg(char *str, unsigned long int &dest) const {
			return ( 0 < sscanf(str,"%lu",&dest) );
		}

		bool scan_arg(char *str, unsigned long long int &dest) const {
			return ( 0 < sscanf(str,"%llu",&dest) );
		}


		bool scan_arg(char *str,   signed char &dest) const {
			return ( 0 < sscanf(str,"%hhd",&dest) );
		}

		bool scan_arg(char *str,   signed short int &dest) const {
			return ( 0 < sscanf(str,"%hd",&dest) );
		}

		bool scan_arg(char *str,   signed int &dest) const {
			return ( 0 < sscanf(str,"%d",&dest) );
		}

		bool scan_arg(char *str,   signed long int &dest) const {
			return ( 0 < sscanf(str,"%ld",&dest) );
		}

		bool scan_arg(char *str,   signed long long int &dest) const {
			return ( 0 < sscanf(str,"%lld",&dest) );
		}



		bool scan_arg(char *str, float &dest) const {
			return ( 0 < sscanf(str,"%f",&dest) );
		}

		bool scan_arg(char *str, double &dest) const {
			return ( 0 < sscanf(str,"%lf",&dest) );
		}


		bool scan_arg(char *str, bool &dest) const{
			if        ( strcmp(str,"false") == 0 ){
				dest = false;
			} else if ( strcmp(str,"true" ) == 0 ){
				dest = true;
			} else {
				return false;
			}
			return true;
		}



		ArgQuery(char* f, char* v) : flag_str(f), value_str(v) {}

		template<typename T>
		void scan_or_fail(T& dest) const{
			if(value_str == NULL) {
				printf("No value provided for flag '-%s'\n", flag_str);
				std::exit(1);
			}
			if( !scan_arg(value_str,dest) ){
				printf("Value string '%s' provided for flag '-%s' "
					"could not be parsed\n",
					value_str,flag_str
				);
				std::exit(1);
			}
		}

		void scan_or_fail(bool& dest) const{
			dest = (value_str != NULL);
		}

		template<typename T>
		ArgVal<T> operator| (T other) {
			if(value_str == NULL){
				return ArgVal<T>(other);
			} else {
				T value;
				scan_or_fail(value);
				return ArgVal<T>(value);
			}
		}

		ArgQuery operator| (ArgQuery other) {
			if(value_str == NULL){
				return other;
			} else {
				return *this;
			}
		}


		template<typename T>
		operator T() const {
			T value;
			scan_or_fail(value);
			return value;
		}

		ArgQuery operator[] (const char* flag_str) {
			return (*this)[(char*)flag_str];
		}

	};


	ArgQuery operator[] (char const* flag_str) {
		return (*this)[(char*)flag_str];
	}

	ArgQuery operator[] (char* flag_str) {
		char* val_str = get_flag_str(flag_str);
		return ArgQuery(flag_str,val_str);
	}

	ArgSet(int c, char** v) : argc(c), argv(v) {}

};



}

}


#endif



