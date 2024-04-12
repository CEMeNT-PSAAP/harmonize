
#ifndef HARMONIZE_CLI
#define HARMONIZE_CLI



namespace util {

namespace cli {



struct GraphStyle {

	int rows;
	int cols;
	const char** tiles;

	GraphStyle (unsigned int r, unsigned int c, const char** t);

};


struct GraphShape {

	int width;
	int height;

	float x_min;
	float x_max;
	float y_min;
	float y_max;

};



extern const char* BlockFillTiles[];
extern const char* Block2x2FillTiles[];
extern const char* Block2x3FillTiles[];
extern const char* AsciiFillTiles[];
extern const char* BrailleFillTiles[25];
extern const GraphStyle BlockFill;
extern const GraphStyle Block2x2Fill;
extern const GraphStyle Block2x3Fill;
extern const GraphStyle AsciiFill;
extern const GraphStyle BrailleFill;


unsigned int lead_char_count(float val);
void cli_graph(float* data, int size, GraphShape shape, GraphStyle style);
void cli_graph(float* data, int size, int width, int height, float low, float high);




struct ArgSet
{

	int    argc;
	char** argv;

	int get_flag_idx(char* flag);
	char* get_flag_str(char* flag);


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


		bool scan_arg(char *str, unsigned char &dest) const;
		bool scan_arg(char *str, unsigned short int &dest) const;
		bool scan_arg(char *str, unsigned int &dest) const;
		bool scan_arg(char *str, unsigned long int &dest) const;
		bool scan_arg(char *str, unsigned long long int &dest) const;
		bool scan_arg(char *str, signed char &dest) const;
		bool scan_arg(char *str, signed short int &dest) const;
		bool scan_arg(char *str, signed int &dest) const;
		bool scan_arg(char *str, signed long int &dest) const;
		bool scan_arg(char *str, signed long long int &dest) const;
		bool scan_arg(char *str, float &dest) const;
		bool scan_arg(char *str, double &dest) const;
		bool scan_arg(char *str, bool &dest) const;

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

		void scan_or_fail(bool& dest) const;

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

		ArgQuery operator| (ArgQuery other);


		template<typename T>
		operator T() const {
			T value;
			scan_or_fail(value);
			return value;
		}

		ArgQuery(char* f, char* v);

	};


	ArgQuery operator[] (char* flag_str);
	ArgQuery operator[] (const char* flag_str);
	ArgSet(int c, char** v);

};



}

}


#endif



