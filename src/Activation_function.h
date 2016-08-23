#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <string>
#include <Eigen/Dense>

#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;

enum activation_function_type { Tanh, HardTanh, Rectifier, Identity, Sigmoid, InvalidFunction };

inline activation_function_type string_to_activation_function (const std::string &s)
{
    if (s == "identity")
        return Identity;
    else if (s == "rectifier")
        return Rectifier;
    else if (s == "tanh")
        return Tanh;
    else if (s == "hardtanh")
        return HardTanh;
    else if (s == "sigmoid")
        return Sigmoid;
    else
        return InvalidFunction;
}

inline std::string activation_function_to_string (activation_function_type f)
{
    if (f == Identity)
        return "identity";
    else if (f == Rectifier)
        return "rectifier";
    else if (f == Tanh)
        return "tanh";
    else if (f == HardTanh)
        return "hardtanh";
    else if (f == Sigmoid)
        return "sigmoid";
    else {
        std::cerr<< "InvalidFunction"<<std::endl;
		exit(1);
	}
}

struct hardtanh_functor {
  precision_type operator() (precision_type x) const { if (x < -1.) return -1.; else if (x > 1.) return 1.; else return x; }
};

struct dhardtanh_functor {
  precision_type operator() (precision_type x) const { return x > -1. && x < 1. ? 1. : 0.; }
};

struct tanh_functor {
  precision_type operator() (precision_type x) const { return std::tanh(x); }
};

struct dtanh_functor {
  precision_type operator() (precision_type x) const { return 1.-x*x; }
};

struct sigmoid_functor {
  precision_type operator() (precision_type x) const { return 1./(1.+std::exp(-x)); }
};

struct dsigmoid_functor {
  precision_type operator() (precision_type x) const { return x*(1.-x); }
};

struct rectifier_functor {
  precision_type operator() (precision_type x) const { return std::max(double(x), 0.); }
};

struct drectifier_functor {
  precision_type operator() (precision_type x) const { return x > 0. ? 1. : 0.; }
};

class Activation_function
{
    int size;
	activation_function_type f;

    public:
        Activation_function() : size(0), f(Rectifier) { }

	void resize(int size) { this->size = size; }
	void set_activation_function(activation_function_type f) { this->f = f; }

	template <typename Engine>
	void initialize(Engine &engine, bool init_normal, precision_type init_range) { }

	int n_inputs () const { return size; }
	int n_outputs () const { return size; }

        template <typename DerivedIn, typename DerivedOut>
	void fProp(const MatrixBase<DerivedIn> &input, const MatrixBase<DerivedOut> &output) const
        {
	    UNCONST(DerivedOut, output, my_output);

		    switch (f)
		    {
			    case Identity: my_output = input; break;
			    case Rectifier: my_output = input.unaryExpr(rectifier_functor()); break;
			    case Tanh: my_output = input.unaryExpr(tanh_functor()); break;
				case Sigmoid: my_output = 1./(1.+(-1*input.array()).exp()); break;//input.unaryExpr(sigmoid_functor()); break;
			    case HardTanh: my_output = input.unaryExpr(hardtanh_functor()); break;
				case InvalidFunction: std::cerr<<"Invalid function"<<std::endl; exit(1); break;
		    }
        }

        template <typename DerivedGOut, typename DerivedGIn, typename DerivedIn, typename DerivedOut>
	void bProp(const MatrixBase<DerivedGOut> &input, 
      	const MatrixBase<DerivedGIn> &output,
		const MatrixBase<DerivedIn> &finput,
       const MatrixBase<DerivedOut> &foutput) const
        {
	    UNCONST(DerivedGIn, output, my_output);

		    switch (f)	
		    {
			    case Identity: my_output = input; break;
			    case Rectifier: my_output = finput.array().unaryExpr(drectifier_functor()) * input.array(); break;
			    case Tanh: my_output = (1.-foutput.array().square()) * input.array(); break; //foutput.array().unaryExpr(dtanh_functor()) * input.array(); break;
				case Sigmoid: my_output = foutput.array()*(1.-foutput.array()) * input.array(); break; //foutput.array().unaryExpr(dsigmoid_functor()) * input.array(); break;
			    case HardTanh: my_output = finput.array().unaryExpr(dhardtanh_functor()) * input.array(); break;
				case InvalidFunction: std::cerr<<"Invalid function"<<std::endl; exit(1); break;
		    }
        }
};

} // namespace nplm

#endif
