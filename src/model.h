#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <string>
#include <boost/random/mersenne_twister.hpp>

#include "neuralClasses.h"
#include "Activation_function.h"

namespace nplm
{

//input_model: words to embeddings
class input_model{
	public:
		int num_hidden, input_vocab_size, input_embedding_dimension;
		input_model():
			num_hidden(0),
			input_vocab_size(0),
			input_embedding_dimension(0){}

		input_model(int num_hidden, 
				int input_vocab_size, 
				int input_embedding_dimension) :
			num_hidden(num_hidden),
			input_vocab_size(input_vocab_size),
			input_embedding_dimension(input_embedding_dimension){}
				
		virtual void resize(int input_vocab_size,
				int input_embedding_dimension,
				int num_hidden)	= 0;

	virtual void initializeGradient(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon) = 0;
	
	virtual void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon) = 0;
	
	virtual void scale(const precision_type scaling_constant) {
		cerr<<"dropout scaling not implemented"<<endl;
	}
	virtual void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							bool norm_clipping,
							precision_type norm_threshold) = 0;
	virtual void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							precision_type grad_scale) {
		cerr<<"write not implemented "<<endl;
		exit(1);
	}							
	virtual void resetGradient() = 0;	
	virtual void write(std::ofstream &file) {
		cerr<<"write not implemented "<<endl;
		exit(1);
	};			
	virtual void read(const string &filename) {
		cerr<<"Read not implemented "<<endl;
		exit(1);
	}
	virtual void readConfig(ifstream config_file) {
		cerr<<"Write config not implemented"<<endl;
		exit(1);
	}
	virtual precision_type getGradSqdNorm() {
		return(0);
	}
};


//the model parameters used for encoder or decoder LSTM nodes
class model {
public:
    Output_word_embeddings output_layer;

  	Linear_layer W_h_to_i, W_h_to_f, W_h_to_c, W_h_to_o; //previous hidden state linked to current input gate, forget gate, output gate and cell
  	Linear_diagonal_layer W_c_to_i, W_c_to_f, W_c_to_o; //previous cell state linked to current input gate, forget gate and output gate
    Hidden_layer i_t, f_t, o_t, tanh_c_prime_t; //input gate, forget gate, output gate and c_prime_t
  	Activation_function tanh_c_t; //c_t
	input_model *input; //input words --> embeddings --> forward results for input gates, forget gates and output gates
	//for feed input
	input_model *feed_input; //attention output embeddings --> forward results for input gates, forget gates and output gates
	
	//for attention
	Linear_layer W_h_to_atten, W_ctx_to_atten; //W_h_to_atten: link decoder LSTM hidden state to attention; W_ctx_to_atten: link source-side context state to attention
	Hidden_layer atten_t; //atten_t outputs attention outputs with non-linear transformation
    
    activation_function_type activation_function;
    int ngram_size, input_vocab_size, output_vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
	bool feed_input_flag;

    model(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension,
        bool share_embeddings):feed_input_flag(false) 
    {

        resize(ngram_size,
            input_vocab_size,
            output_vocab_size,
            input_embedding_dimension,
            num_hidden,
            output_embedding_dimension);
    }
	int get_hidden() {return num_hidden;}
	
	void set_input(input_model &input){this->input = &input;}
	void set_feed_input_flag(const bool flag){this->feed_input_flag = flag;}
	void set_feed_input(input_model &feed_input){this->feed_input = &feed_input;}
	
    model() : input(NULL),
			feed_input(NULL),
			ngram_size(1), 
			feed_input_flag(false),
            activation_function(Rectifier)
        {
        }
		
	precision_type getGradSqdNorm();

    void resize(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension);

    void initializeGradient(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
		precision_type init_output_bias,
        precision_type init_forget_bias,
        string &parameter_udpate,
        precision_type adagrad_epsilon);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
		precision_type init_output_bias,
        precision_type init_forget_bias,
        string &parameter_udpate,
        precision_type adagrad_epsilon);

    void set_activation_function(activation_function_type f)
    {
        activation_function = f;
    }
	
    void set_activation_functions()
    {
		//all gates use sigmoid activation function
		o_t.set_activation_function(Sigmoid);

		f_t.set_activation_function(Sigmoid);	

		i_t.set_activation_function(Sigmoid);		
	
		//others use tanh activation function
		tanh_c_prime_t.set_activation_function(Tanh);
	
		tanh_c_t.set_activation_function(Tanh);

		//for attention layer
		atten_t.set_activation_function(Tanh);
    }

	//read the model from files
    void read(const std::string &filename);
	void read(const string &filename, int &start_word, int &end_word);
    void read(const std::string &filename, std::vector<std::string> &words);
    void read(const std::string &filename, std::vector<std::string> &input_words, std::vector<std::string> &output_words);
	//write the model to files
    void write(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> &words);
    void write(const std::string &filename, const std::vector<std::string> &input_words, const std::vector<std::string> &output_words);
	//update all the network parameters in the model
	void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
				  		precision_type momentum,
						precision_type L2_reg,
						bool norm_clipping,
						precision_type norm_threshold);
	void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
				  		precision_type momentum,
						precision_type L2_reg,
						precision_type grad_scale);						
	void resetGradient();
	void scale(precision_type scaling_constant);
 private:
    void readConfig(std::ifstream &config_file);
    void readConfig(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> *input_pwords, const std::vector<std::string> *output_pwords);
};


//google_input_model: words --> embeddings --> forward results for next gates
class google_input_model : public input_model {
	public:
		Input_word_embeddings input_layer; //words --> embeddings
		Linear_layer W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o; //weights link word embeddings to different gates to generate forward results as gate inputs
 
		int num_hidden, input_vocab_size, input_embedding_dimension;

		google_input_model(int num_hidden,
				int input_vocab_size,
				int input_embedding_dimension):input_model(num_hidden,input_vocab_size,input_embedding_dimension)  
		{
		}
	
		google_input_model() : 
			input_model() 
        {
        }
	
		void resize(int input_vocab_size,
				int input_embedding_dimension,
				int num_hidden);

		void initializeGradient(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);	
		
		void initialize(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);	
		
		//update weights and word embeddings
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				bool norm_clipping,
				precision_type norm_threshold);
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				precision_type grad_scale);									
		void resetGradient();
		void scale(const precision_type scaling_constant);
		void write(ofstream &file);
		void read(const string &filename);
		void readConfig(ifstream &config_file);
		precision_type getGradSqdNorm();
};

//hidden_to_hidden_input_model: embeddings --> forward results for different gates (e.g. attention output --> forward results for gates)
class hidden_to_hidden_input_model : public input_model {
	public:
		Linear_layer W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o; //weights link embeddings to gates

		int num_hidden, input_vocab_size, input_embedding_dimension;
		precision_type norm_threshold;

		hidden_to_hidden_input_model(int num_hidden,
				int input_vocab_size,
				int input_embedding_dimension,
				precision_type norm_threshold):input_model(num_hidden,input_vocab_size,input_embedding_dimension)  
		{
			this->norm_threshold = norm_threshold;
		}
	
		hidden_to_hidden_input_model() : 
			input_model()
		{
			this->norm_threshold = 0.5;
        }

		void resize(int input_vocab_size,
				int input_embedding_dimension,
				int num_hidden);

		void initializeGradient(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);	
		
		void initialize(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);	
		
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				bool norm_clipping,
				precision_type norm_threshold);
					
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				precision_type grad_scale);

		void resetGradient();
		void write(ofstream &file);
		void readConfig(ifstream &config_file);
		void read(const string &filename);
};


class standard_input_model : public input_model{
	public:
		Input_word_embeddings W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
		Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>  W_x_to_i_embedding_matrix, 
			W_x_to_f_embedding_matrix, 
			W_x_to_c_embedding_matrix, 
			W_x_to_o_embedding_matrix; 
 
		standard_input_model(int num_hidden,
				int input_vocab_size,
				int input_embedding_dimension) :input_model(num_hidden,input_vocab_size,input_embedding_dimension) 
		{
		}
	
		standard_input_model() : 
			input_model()
        {
        }

		void resize(int input_vocab_size,
				int input_embedding_dimension,
				int num_hidden);

		void initializeGradient(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);
			
		void initialize(boost::random::mt19937 &init_engine,
				bool init_normal,
				precision_type init_range,
				string &parameter_udpate,
				precision_type adagrad_epsilon);
			
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				bool norm_clipping,
				precision_type norm_threshold);
		void resetGradient(){}
};

} //namespace nplm

#endif
