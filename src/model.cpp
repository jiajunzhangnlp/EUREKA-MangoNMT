#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <boost/lexical_cast.hpp>

#include "model.h"
#include "param.h"

using namespace std;
using namespace boost;
using namespace boost::random;

namespace nplm
{

void model::resize(int ngram_size,
    int input_vocab_size,
    int output_vocab_size,
    int input_embedding_dimension,
    int num_hidden,
    int output_embedding_dimension)
{
    output_layer.resize(output_vocab_size, output_embedding_dimension);
    this->ngram_size = ngram_size;
    this->input_vocab_size = input_vocab_size;
    this->output_vocab_size = output_vocab_size;
    this->input_embedding_dimension = input_embedding_dimension;
    this->num_hidden = num_hidden;
    this->output_embedding_dimension = output_embedding_dimension;
	
	//Resizing weight matrices
	W_h_to_c.resize(num_hidden,num_hidden);
	W_h_to_f.resize(num_hidden,num_hidden);
	W_h_to_i.resize(num_hidden,num_hidden);
	W_h_to_o.resize(num_hidden,num_hidden);

	//for attention
	W_h_to_atten.resize(num_hidden,num_hidden);
	W_ctx_to_atten.resize(num_hidden,num_hidden);
	
	W_c_to_f.resize(num_hidden);
	W_c_to_i.resize(num_hidden);
	W_c_to_o.resize(num_hidden);
	
	//Resizing weight matrices and activation functions
	o_t.resize(num_hidden);
	f_t.resize(num_hidden);
	i_t.resize(num_hidden);
	tanh_c_prime_t.resize(num_hidden);
	tanh_c_t.resize(num_hidden);
	atten_t.resize(num_hidden);
}

 void model::initializeGradient(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
	precision_type init_output_bias,
    precision_type init_forget_bias,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
    output_layer.initializeGradient(init_engine,
        init_normal,
        init_range,
        init_output_bias,
        parameter_update,
        adagrad_epsilon);

	W_h_to_c.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
		
	W_h_to_i.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);	
		
	W_h_to_o.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
	
	W_h_to_f.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);

	//for attention
	W_h_to_atten.initializeGradient(init_engine,
			init_normal,
			init_range,
			parameter_update,
			adagrad_epsilon);

	W_ctx_to_atten.initializeGradient(init_engine,
			init_normal,
			init_range,
			parameter_update,
			adagrad_epsilon);

	#ifdef PEEP
	W_c_to_i.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);	
	W_c_to_o.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
	W_c_to_f.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);		
	#endif
		
	o_t.initializeGradient(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	o_t.set_activation_function(Sigmoid);
	
	f_t.initializeGradient(init_engine,
        init_normal,
        init_range,
		init_forget_bias,
        parameter_update,
        adagrad_epsilon);
	f_t.set_activation_function(Sigmoid);	
	
	i_t.initializeGradient(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	i_t.set_activation_function(Sigmoid);		
	
	tanh_c_prime_t.initializeGradient(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	tanh_c_prime_t.set_activation_function(Tanh);
	
	tanh_c_t.set_activation_function(Tanh);

	atten_t.initializeGradient(init_engine,
			init_normal,
			init_range,
			0.,
			parameter_update,
			adagrad_epsilon);
	atten_t.set_activation_function(Tanh);
}
 

void model::initialize(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
	precision_type init_output_bias,
    precision_type init_forget_bias,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
    output_layer.initialize(init_engine,
        init_normal,
        init_range,
        init_output_bias,
        parameter_update,
        adagrad_epsilon);

	W_h_to_c.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
		
	W_h_to_i.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);	
		
	W_h_to_o.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
	
	W_h_to_f.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);

	//for attention
	W_h_to_atten.initialize(init_engine,
			init_normal,
			init_range,
			parameter_update,
			adagrad_epsilon);

	W_ctx_to_atten.initialize(init_engine,
			init_normal,
			init_range,
			parameter_update,
			adagrad_epsilon);

	#ifdef PEEP
	W_c_to_i.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);	
	W_c_to_o.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
	W_c_to_f.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);		
	#endif
		
	o_t.initialize(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	o_t.set_activation_function(Sigmoid);
	
	f_t.initialize(init_engine,
        init_normal,
        init_range,
		init_forget_bias,
        parameter_update,
        adagrad_epsilon);
	f_t.set_activation_function(Sigmoid);	
	
	i_t.initialize(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	i_t.set_activation_function(Sigmoid);		
	
	tanh_c_prime_t.initialize(init_engine,
        init_normal,
        init_range,
		0.,
        parameter_update,
        adagrad_epsilon);
	tanh_c_prime_t.set_activation_function(Tanh);
	
	tanh_c_t.set_activation_function(Tanh);

	atten_t.initialize(init_engine,
			init_normal,
			init_range,
			0.,
			parameter_update,
			adagrad_epsilon);
	atten_t.set_activation_function(Tanh);
}

void model::readConfig(ifstream &config_file)
{
	string line;
	vector<string> fields;
	int ngram_size, vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
	activation_function_type activation_function = this->activation_function;
	while (getline(config_file, line) && line != "")
	{
		splitBySpace(line, fields);
		if (fields[0] == "input_vocab_size")
			input_vocab_size = lexical_cast<int>(fields[1]);
		else if (fields[0] == "output_vocab_size")
			output_vocab_size = lexical_cast<int>(fields[1]);
		else if (fields[0] == "num_hidden") {
			num_hidden = lexical_cast<int>(fields[1]);
			input_embedding_dimension = num_hidden;
			output_embedding_dimension = num_hidden;
		}
		else if (fields[0] == "version")
		{
			int version = lexical_cast<int>(fields[1]);
			if (version != 1)
			{
				cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
				exit(1);
			}
		}
		else {
		}
	}

	resize(1,
			input_vocab_size,
			output_vocab_size,
			input_embedding_dimension,
			num_hidden,
			output_embedding_dimension);
	//setting all the activation functions
	set_activation_functions();
}

void model::readConfig(const string &filename)
{
    ifstream config_file(filename.c_str());
    if (!config_file)
    {
        cerr << "error: could not open config file " << filename << endl;
		exit(1);
    }
    readConfig(config_file);
    config_file.close();
}
 
void model::read(const string &filename)
{
    vector<string> input_words;
    vector<string> output_words;
    read(filename, input_words, output_words);
}

void model::read(const string &filename, vector<string> &words)
{
    vector<string> output_words;
    read(filename, words, output_words);
}

void model::read(const string &filename, vector<string> &input_words, vector<string> &output_words)
{
    ifstream file(filename.c_str());
    if (!file) throw runtime_error("Could not open file " + filename);
    
    param myParam;
    string line;
    
    while (getline(file, line))
    {	
	if (line == "\\config")
	{
	    readConfig(file);
	}

	else if (line == "\\vocab")
	{
	    input_words.clear();
	    readWordsFile(file, input_words);
	    output_words = input_words;
	}

	else if (line == "\\input_vocab")
	{
	    input_words.clear();
	    readWordsFile(file, input_words);
	}

	else if (line == "\\output_vocab")
	{
	    output_words.clear();
	    readWordsFile(file, output_words);
	}
	else if (line == "\\W_h_to_i")
	    W_h_to_i.read_weights(file);
	else if (line == "\\W_h_to_f")
	    W_h_to_f.read_weights(file);
	else if (line == "\\W_h_to_o")
	    W_h_to_o.read_weights(file);
	else if (line == "\\W_h_to_c")
	    W_h_to_c.read_weights(file);
	else if (line == "\\W_c_to_i")
	    W_c_to_i.read_weights(file);
	else if (line == "\\W_c_to_f")
	    W_c_to_f.read_weights(file);
	else if (line == "\\W_c_to_o")
	    W_c_to_o.read_weights(file);
	//for attetnion
	else if (line == "\\W_h_to_atten")
		W_h_to_atten.read_weights(file);
	else if (line == "\\W_ctx_to_atten")
		W_ctx_to_atten.read_weights(file);
	else if (line == "\\attention_biases")
		atten_t.read_biases(file);
	else if (line == "\\input_gate_biases")
	    i_t.read_biases(file);	
	else if (line == "\\forget_gate_biases")
	    f_t.read_biases(file);	
	else if (line == "\\tanh_c_prime_t_gate_biases")
	    tanh_c_prime_t.read_biases(file);
	else if (line == "\\output_gate_biases")
	    o_t.read_biases(file);	
	else if (line == "\\output_weights")
	    output_layer.read_weights(file);
	else if (line == "\\output_biases")
	    output_layer.read_biases(file);	
	else if (line == "\\feed_input_weights")
		feed_input_flag = true;
	else if (line == "\\end")
	    break;
	else if (line == "")
	    continue;
	else
	{
	    // skip over section
	    while (getline(file, line) && line != "") { }
	}
    }
    file.close();
}

void model::write(const string &filename, const vector<string> &input_words, const vector<string> &output_words)
{ 
    write(filename, &input_words, &output_words);
}

void model::write(const string &filename, const vector<string> &words)
{ 
    write(filename, &words, NULL);
}

void model::write(const string &filename) 
{ 
    write(filename, NULL, NULL);
}

void model::write(const string &filename, const vector<string> *input_pwords, const vector<string> *output_pwords)
{
    ofstream file(filename.c_str());

	file << std::setprecision(15);
    if (!file) throw runtime_error("Could not open file " + filename);
    
    file << "\\config" << endl;
    file << "version 1" << endl;
    file << "input_vocab_size " << input_vocab_size << endl;
    file << "output_vocab_size " << output_vocab_size << endl;
    file << "num_hidden " << num_hidden << endl;
    file << endl;
    
    if (input_pwords)
    {
        file << "\\input_vocab" << endl;
	writeWordsFile(*input_pwords, file);
	file << endl;
    }

    if (output_pwords)
    {
        file << "\\output_vocab" << endl;
	writeWordsFile(*output_pwords, file);
	file << endl;
    }
	
	//First writing the input
	input->write(file);

	if( feed_input_flag ){
		file << "\\feed_input_weights" << endl;
		file << endl;
		feed_input->write(file);
	}
		
    file << "\\W_h_to_i" << endl;
    W_h_to_i.write_weights(file);
    file << endl;

    file << "\\W_h_to_f" << endl;
    W_h_to_f.write_weights(file);
    file << endl;

    file << "\\W_h_to_o" << endl;
    W_h_to_o.write_weights(file);
    file << endl;

    file << "\\W_h_to_c" << endl;
    W_h_to_c.write_weights(file);
    file << endl;
	
    file << "\\W_c_to_i" << endl;
    W_c_to_i.write_weights(file);
    file << endl;

    file << "\\W_c_to_f" << endl;
    W_c_to_f.write_weights(file);
    file << endl;

    file << "\\W_c_to_o" << endl;
    W_c_to_o.write_weights(file);
    file << endl;

	//for attention
	file << "\\W_h_to_atten" << endl;
	W_h_to_atten.write_weights(file);
	file << endl;

	file << "\\W_ctx_to_atten" << endl;
	W_ctx_to_atten.write_weights(file);
	file << endl;

	file << "\\attention_biases" << endl;
	atten_t.write_biases(file);
	file << endl;
 
    file << "\\input_gate_biases" << endl;
    i_t.write_biases(file);
    file << endl;

    file << "\\forget_gate_biases" << endl;
    f_t.write_biases(file);
    file << endl;

    file << "\\tanh_c_prime_t_gate_biases" << endl;
    tanh_c_prime_t.write_biases(file);
    file << endl;

    file << "\\output_gate_biases" << endl;
    o_t.write_biases(file);
    file << endl;

    file << "\\output_biases" << endl;
    output_layer.write_biases(file);
    file << endl;
					   
    file << "\\output_weights" << endl;
    output_layer.write_weights(file);
    file << endl;
    
    file << "\\end" << endl;
    file.close();
}

void model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					bool norm_clipping,
					precision_type norm_threshold){
						// updating the rest of the parameters
		
						//updating params for weights out of hidden layer 
						W_h_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
				 		W_h_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
				  		W_h_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
				   		W_h_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
				   		W_h_to_atten.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
				   		W_ctx_to_atten.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						#ifdef PEEP
															
									
						//updating params for weights out of cell
						W_c_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_c_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_c_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);	
						#endif			


																										
						o_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						f_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						i_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);	
						tanh_c_prime_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);	
															
						//for attention
						atten_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);	

						//Updaging the parameters of the input model
						input->updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);				

						//Updaging the parameters of the feed input model
						if( feed_input_flag )
							feed_input->updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);				
}

void model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					precision_type grad_scale) {
						// updating the rest of the parameters
		
						//updating params for weights out of hidden layer 
						W_h_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
				 		W_h_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
				  		W_h_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
				   		W_h_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						//for attention
				   		W_h_to_atten.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
				   		W_ctx_to_atten.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						#ifdef PEEP
															
									
						//updating params for weights out of cell
						W_c_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_c_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_c_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);	
						#endif			


																										
						o_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						f_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						i_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);	
						tanh_c_prime_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);	
															
						//for attention
						atten_t.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);	
						//Updaging the parameters of the input model
						input->updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);				

						//Updaging the parameters of the feed input model
						if( feed_input_flag )
							feed_input->updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);				
}

precision_type model::getGradSqdNorm(){
		precision_type model_grad_squared_norm = 0.;
		model_grad_squared_norm += W_h_to_o.getGradSqdNorm() + 
				 		W_h_to_f.getGradSqdNorm() +
				  		W_h_to_i.getGradSqdNorm() +
				   		W_h_to_c.getGradSqdNorm() +
				   		W_h_to_atten.getGradSqdNorm() +
				   		W_ctx_to_atten.getGradSqdNorm() +
						#ifdef PEEP																				
						//updating params for weights out of cell
						W_c_to_f.getGradSqdNorm() + 
						W_c_to_i.getGradSqdNorm() +
						W_c_to_o.getGradSqdNorm() +	
						#endif																						
						o_t.getGradSqdNorm() +
						f_t.getGradSqdNorm() +
						i_t.getGradSqdNorm() +
						tanh_c_prime_t.getGradSqdNorm() +
						atten_t.getGradSqdNorm() +
						input->getGradSqdNorm();		
		if( feed_input_flag )
			model_grad_squared_norm += feed_input->getGradSqdNorm();
		return(model_grad_squared_norm);
	}


void model::resetGradient(){
	output_layer.resetGradient();
	// updating the rest of the parameters
	
	//updating params for weights out of hidden layer 
	W_h_to_o.resetGradient();
	W_h_to_f.resetGradient();
	W_h_to_i.resetGradient();
	W_h_to_c.resetGradient();
	//for attention
	W_h_to_atten.resetGradient();
	W_ctx_to_atten.resetGradient();

	#ifdef PEEP
	//updating params for weights out of cell
	W_c_to_f.resetGradient();
	W_c_to_i.resetGradient();
	W_c_to_o.resetGradient();				
	#endif

	//Computing gradients of the paramters
	o_t.resetGradient();
	f_t.resetGradient();
	i_t.resetGradient();	
	tanh_c_prime_t.resetGradient();	
							
	//for attention
	atten_t.resetGradient();
	//The gradients of the input layer are being reset in update params sinc the gradient is sparse
	//Derivatives of the input embeddings							
	input->resetGradient();

	if( feed_input_flag )
		feed_input->resetGradient();
}	

//Scaling weight if using dropout at decode time
void model::scale(precision_type scaling_constant) {
	output_layer.scale(scaling_constant);
}

void google_input_model::resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden)
{
    input_layer.resize(input_vocab_size, input_embedding_dimension, 1); // the input is always dimension 1 now.

    this->input_vocab_size = input_vocab_size;
    this->input_embedding_dimension = input_embedding_dimension;
    this->num_hidden = num_hidden;
	
	W_x_to_c.resize(num_hidden,num_hidden);
	W_x_to_o.resize(num_hidden,num_hidden);
	W_x_to_f.resize(num_hidden,num_hidden);
	W_x_to_i.resize(num_hidden,num_hidden);
}

precision_type google_input_model::getGradSqdNorm() {
	return (
    input_layer.getGradSqdNorm() +		
		W_x_to_c.getGradSqdNorm() +
		W_x_to_i.getGradSqdNorm() +
		W_x_to_o.getGradSqdNorm() +
		W_x_to_f.getGradSqdNorm()) ;			
}

void google_input_model::initializeGradient(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
    input_layer.initializeGradient(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
		
		W_x_to_c.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}
 
void google_input_model::initialize(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
    input_layer.initialize(init_engine,
        init_normal,
        init_range,
        parameter_update,
        adagrad_epsilon);
		
		W_x_to_c.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}

void google_input_model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					bool norm_clipping,
					precision_type norm_threshold){
						
						input_layer.updateParams(learning_rate,
																	current_minibatch_size,
																	momentum,
																	L2_reg,
																	norm_clipping,
																	norm_threshold);

						//Error derivatives for the linear layers from the input word embeddings
						W_x_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
}

void google_input_model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					precision_type grad_scale){
						
						input_layer.updateParams(learning_rate,
																	current_minibatch_size,
																	momentum,
																	L2_reg,
																	grad_scale);

						//Error derivatives for the linear layers from the input word embeddings
						W_x_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
}

void google_input_model::resetGradient(){

	//Error derivatives for the input word embeddings
	W_x_to_c.resetGradient();
	W_x_to_o.resetGradient();
	W_x_to_f.resetGradient();
	W_x_to_i.resetGradient();
}	


void google_input_model::write(std::ofstream &file)
{
	file <<"\\input_weights"<<endl;
    input_layer.write(file);
    file << endl;

    file << "\\W_x_to_i" << endl;
    W_x_to_i.write_weights(file);
    file << endl;

    file << "\\W_x_to_f" << endl;
    W_x_to_f.write_weights(file);
    file << endl;
	    
    file << "\\W_x_to_o" << endl;
    W_x_to_o.write_weights(file);
    file << endl;

    file << "\\W_x_to_c" << endl;
    W_x_to_c.write_weights(file);
    file << endl;
}


void google_input_model::readConfig(ifstream &config_file)
	{
	    string line;
	    vector<string> fields;
	    int input_vocab_size, input_embedding_dimension, num_hidden;
	    while (getline(config_file, line) && line != "")
	    {
	       splitBySpace(line, fields);

		if (fields[0] == "input_vocab_size")
		    input_vocab_size = lexical_cast<int>(fields[1]);
		else if (fields[0] == "num_hidden") {
		    num_hidden = lexical_cast<int>(fields[1]);
			input_embedding_dimension = num_hidden;
		}
		else if (fields[0] == "version")
		{
		    int version = lexical_cast<int>(fields[1]);
		    if (version != 1)
		    {
			cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
			exit(1);
		    }
		}
		else {
		    //cerr << "warning: unrecognized field in config: " << fields[0] << endl;
		}
	  }
	    resize(input_vocab_size,
		    input_embedding_dimension,
		    num_hidden);	
}


void google_input_model::read(const string &filename)
{
	ifstream file(filename.c_str());
	if (!file) throw runtime_error("Could not open file " + filename);

	param myParam;
	string line;

	while (getline(file, line))
	{	
		if (line == "\\config")
		{
			readConfig(file);
			//cerr<<"About to read the weights in google input model"<<endl;
		}
		else if (line == "\\W_x_to_c") 
			W_x_to_c.read_weights(file);	
		else if (line == "\\W_x_to_i")
			W_x_to_i.read_weights(file);
		else if (line == "\\W_x_to_f")
			W_x_to_f.read_weights(file);
		else if (line == "\\W_x_to_o")
			W_x_to_o.read_weights(file);
		else if (line == "\\input_weights") 
			input_layer.read(file);
		else if (line == "\\end")
			break;
		else if (line == "")
			continue;
		else
		{
			//cerr << "warning: unrecognized section: " << line << endl;
			// skip over section
			while (getline(file, line) && line != "") { }
		}
	}
	file.close();
}

//Scaling weight if using dropout at decode time
void google_input_model::scale(precision_type scaling_constant) {
	W_x_to_c.scale(scaling_constant);
	W_x_to_o.scale(scaling_constant);
	W_x_to_f.scale(scaling_constant);
	W_x_to_i.scale(scaling_constant);	
}

void hidden_to_hidden_input_model::resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden)
{
    this->input_vocab_size = input_vocab_size;
    this->input_embedding_dimension = input_embedding_dimension;
    this->num_hidden = num_hidden;
	

	W_x_to_c.resize(num_hidden,num_hidden);
	W_x_to_o.resize(num_hidden,num_hidden);
	W_x_to_f.resize(num_hidden,num_hidden);
	W_x_to_i.resize(num_hidden,num_hidden);
}

void hidden_to_hidden_input_model::initializeGradient(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
		W_x_to_c.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}


  
void hidden_to_hidden_input_model::initialize(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
		W_x_to_c.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}

void hidden_to_hidden_input_model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					bool norm_clipping,
					precision_type norm_threshold){
						

						//Error derivatives for the linear layers from the input word embeddings
						W_x_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);

						W_x_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
}

void hidden_to_hidden_input_model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					precision_type grad_scale){
						//Error derivatives for the linear layers from the input word embeddings
						W_x_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
						W_x_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															grad_scale);
}

void hidden_to_hidden_input_model::resetGradient(){
	//Error derivatives for the input word embeddings
	W_x_to_c.resetGradient();
	W_x_to_o.resetGradient();
	W_x_to_f.resetGradient();
	W_x_to_i.resetGradient();
}	

void hidden_to_hidden_input_model::write(std::ofstream &file)
{
    file << "\\W_x_to_i_feed" << endl;
    W_x_to_i.write_weights(file);
    file << endl;

    file << "\\W_x_to_f_feed" << endl;
    W_x_to_f.write_weights(file);
    file << endl;
	    
    file << "\\W_x_to_o_feed" << endl;
    W_x_to_o.write_weights(file);
    file << endl;

    file << "\\W_x_to_c_feed" << endl;
    W_x_to_c.write_weights(file);
    file << endl;
}

void hidden_to_hidden_input_model::readConfig(ifstream &config_file)
	{
	    string line;
	    vector<string> fields;
	    int input_vocab_size, input_embedding_dimension, num_hidden;
	    while (getline(config_file, line) && line != "")
	    {
	       splitBySpace(line, fields);

		if (fields[0] == "input_vocab_size")
		    input_vocab_size = lexical_cast<int>(fields[1]);
		else if (fields[0] == "num_hidden") {
		    num_hidden = lexical_cast<int>(fields[1]);
			input_embedding_dimension = num_hidden;
		}
		else if (fields[0] == "version")
		{
		    int version = lexical_cast<int>(fields[1]);
		    if (version != 1)
		    {
			cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
			exit(1);
		    }
		}
		else {
		    //cerr << "warning: unrecognized field in config: " << fields[0] << endl;
		}
	  }
	    resize(input_vocab_size,
		    input_embedding_dimension,
		    num_hidden);	
}


void hidden_to_hidden_input_model::read(const string &filename)
{
	ifstream file(filename.c_str());
	if (!file) throw runtime_error("Could not open file " + filename);

	param myParam;
	string line;

	while (getline(file, line))
	{	
		if (line == "\\config")
		{
			readConfig(file);
			//cerr<<"About to read the weights in google input model"<<endl;
		}
		else if (line == "\\W_x_to_c_feed") 
			W_x_to_c.read_weights(file);	
		else if (line == "\\W_x_to_i_feed")
			W_x_to_i.read_weights(file);
		else if (line == "\\W_x_to_f_feed")
			W_x_to_f.read_weights(file);
		else if (line == "\\W_x_to_o_feed")
			W_x_to_o.read_weights(file);
		else if (line == "\\end")
			break;
		else if (line == "")
			continue;
		else
		{
			//cerr << "warning: unrecognized section: " << line << endl;
			// skip over section
			while (getline(file, line) && line != "") { }
		}
	}
	file.close();
}

void standard_input_model::resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden)
{
    this->input_vocab_size = input_vocab_size;
    this->input_embedding_dimension = input_embedding_dimension;
    this->num_hidden = num_hidden;

	W_x_to_c.resize(input_vocab_size, input_embedding_dimension, 1);
	W_x_to_o.resize(input_vocab_size, input_embedding_dimension, 1);
	W_x_to_f.resize(input_vocab_size, input_embedding_dimension, 1);
	W_x_to_i.resize(input_vocab_size, input_embedding_dimension, 1);
}

void standard_input_model::initializeGradient(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
		W_x_to_c.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initializeGradient(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}
 
void standard_input_model::initialize(mt19937 &init_engine,
    bool init_normal,
    precision_type init_range,
    string &parameter_update,
    precision_type adagrad_epsilon)
{
		W_x_to_c.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
	
		W_x_to_i.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);	
		
		W_x_to_o.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);
		
		W_x_to_f.initialize(init_engine,
	        init_normal,
	        init_range,
	        parameter_update,
	        adagrad_epsilon);			
}

void standard_input_model::updateParams(precision_type learning_rate,
 					int current_minibatch_size,
			  		precision_type momentum,
					precision_type L2_reg,
					bool norm_clipping,
					precision_type norm_threshold){
						
						//Error derivatives for the linear layers from the input word embeddings
						W_x_to_c.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_o.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_f.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
						W_x_to_i.updateParams(learning_rate,
															current_minibatch_size,
															momentum,
															L2_reg,
															norm_clipping,
															norm_threshold);
}

} // namespace nplm
