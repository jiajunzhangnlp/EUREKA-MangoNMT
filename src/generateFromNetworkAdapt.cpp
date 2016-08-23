//#define EIGEN_RUNTIME_NO_MALLOC
#include <ctime>
#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>


#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

#include "define.h"
#include "model.h"
#include "propagator.h"
#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "multinomial.h"
#include "vocabulary.h"


using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost;
using namespace boost::random;

using namespace nplm;

namespace ip = boost::interprocess;
typedef unordered_map<Matrix<int,Dynamic,1>, precision_type> vector_map;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;


template<typename VectorDerivedO, 
			typename VectorDerivedF, 
			typename VectorDerivedI, 
			typename VectorDerivedH, 
			typename VectorDerivedC>
void writeStates(ofstream &hidden_states_file,
					const vector<VectorDerivedH >   &internal_h_t,
					const vector<VectorDerivedC>   &internal_c_t,
					const vector<VectorDerivedF>   &internal_f_t,
					const vector<VectorDerivedI>   &internal_i_t,
					const vector<VectorDerivedO>   &internal_o_t,
					const int minibatch_index,
					const int word_index){
	hidden_states_file<<"h_t: ";
	writeMatrix(internal_h_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"c_t: ";
	writeMatrix(internal_c_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"o_t: ";
	writeMatrix(internal_o_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"f_t: ";
	writeMatrix(internal_f_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"i_t: ";
	writeMatrix(internal_i_t[minibatch_index].col(word_index).transpose(),hidden_states_file);			
}

int main(int argc, char** argv)
{ 
	srand (time(NULL));
	setprecision(16);
    ios::sync_with_stdio(false);
    bool randomize, tagging_mode;
    param myParam;
	int arg_output_start_symbol;
	int arg_output_end_symbol;
	string arg_predicted_sequence_file, arg_hidden_states_file;
	bool arg_greedy;
	bool arg_stochastic;
	bool arg_score;
	bool arg_run_lm;
	bool arg_carry_states;
	bool generate_hidden_states = 0;
	bool do_beam_search;
	int arg_beam_size = 0;
	bool arg_reverse_input;
	precision_type arg_min_output_ratio = 1.0;
	precision_type arg_max_output_ratio = 2.0;
	int arg_num_similar_samples = 0;
	precision_type arg_similarity_threshold = 0.2;
	
    try {
      // program options //
      CmdLine cmd("Do search and scoring with a trained LSTM sequence to sequence model or a LSTM language model.", ' ' , "0.1\n","");

      ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 100.", false, 100, "int", cmd);

      ValueArg<precision_type> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "precision_type", cmd);

      ValueArg<precision_type> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 1.", false, 1., "precision_type", cmd);

      ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: log.", false, "log", "string", cmd);

      ValueArg<precision_type> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.1", false, 0.1, "precision_type", cmd);

      ValueArg<precision_type> init_forget("", "init_forget", "value to initialize the bias of the forget gate. Default: 0", false, 0, "precision_type", cmd);

      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0. \
		  (initialize from a uniform distribution)", false, 0, "bool", cmd);

      ValueArg<string> training_sent_file("", "training_sent_file", "Training sentences file" , true, "", "string", cmd);

	  ValueArg<bool> norm_clipping("", "norm_clipping", "Do you want to do norm clipping or gradient clipping. 1 = norm cilpping, \n \
		  			0 = gradient clipping. Default: 1.", false, 1, "bool", cmd);

	  ValueArg<precision_type> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "precision_type", cmd);

	  ValueArg<precision_type> similarity_threshold("", "similarity_threshold", "Threshold for sentence similarity. Default 0.3", false, 0.3, "precision_type", cmd);

	  ValueArg<precision_type> fixed_partition_function("", "fixed_partition_function", "Fixed log normalization constant value. Default: 0.", false, 0., "precision_type", cmd);

	  ValueArg<int> num_similar_samples("", "num_similar_samples", "Number of similar samples for each test sentence we need to fine-tune the model. Default: 64.", false, 64, "int", cmd);

      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 1.", false, 1, "int", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
 
	  ValueArg<string> testing_sent_file("", "testing_sent_file", "Input sentences file." , true, "", "string", cmd);

      ValueArg<string> encoder_model_file("", "encoder_model_file", "Encoder Model file. Default: empty", false, "", "string", cmd);

      ValueArg<string> encoder_reverse_model_file("", "encoder_reverse_model_file", "Encoder Reverse Model file. Default: empty", false, "", "string", cmd);

	  ValueArg<string> decoder_model_file("", "decoder_model_file", "Decoder Model file. Default: empty", true, "", "string", cmd);

	  ValueArg<string> predicted_sequence_file("", "predicted_sequence_file", "Predicted sequences file." , false, "", "string", cmd);

	  ValueArg<bool> greedy("", "greedy", "If yes, then the output will be generated greedily \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	

	  ValueArg<bool> stochastic("", "stochastic", "If yes, then the output will be generated stochastically \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	

	  ValueArg<int> beam_size("", "beam_size", "Do beam search with specified beam size (>=1). Default: 0 (No beam search). \n \
	Note: Please supply an argument to --predicted_sequence_file where the results of beam search will be saved.", false, 0, "int", cmd);		  

	  ValueArg<bool> score("", "score", "If yes, then the program will compute the log probability of output given input \n \
		  or probability of sentence if run as a language model. Default: 0 = no. \n", false, 0, "bool", cmd);	  	  	  

	  ValueArg<bool> run_lm("", "run_lm", "Run as a language model, \n \
		  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);	

	  ValueArg<bool> carry_states("", "carry_states", "Carry the hidden states from one minibatch to another. This option is for \n \
		  			language models only. Carrying hidden states over can improve perplexity. If it was used during training, then it MUST \n \
					be used at testing. 1 = yes. Default: 0 (Do not carry hidden states).", false, 0, "bool", cmd);		  	  

	  ValueArg<bool> reverse_input("", "reverse", "Reverse the input sentence before decoding. \n \
		  			It should be set to the same value as used during training. \n \
		  			1 = yes. Default: 0 (No reversing).", false, 0, "bool", cmd);  

	  ValueArg<string> hidden_states_file("", "hidden_states_file", "Dump the hidden states in this file. Will currently only \
		  generate the forced decode trace. That is, it needs both input and output pairs. Default: empty", false, "", "string", cmd);	  

	  ValueArg<precision_type> min_output_ratio("", "min_output_ratio", "Control the output length at least min_output_ratio*input_length\
		  in order to avoid too short translation. Default: 1.0", false, 1.0, "precision_type", cmd);	  

	  ValueArg<precision_type> max_output_ratio("", "max_output_ratio", "Control the output length at most max_output_ratio*input_length\
		  in order to avoid too long translation. Default: 2.0", false, 2.0, "precision_type", cmd);	  

	  ValueArg<precision_type> dropout_probability("", "dropout_probability", "Dropout probability. Default 0: No dropout", false,0., "precision_type", cmd);

      cmd.parse(argc, argv);

      myParam.num_noise_samples = num_noise_samples.getValue();
      myParam.L2_reg = L2_reg.getValue();
      myParam.learning_rate = learning_rate.getValue();
      myParam.loss_function = loss_function.getValue();
      myParam.training_sent_file = training_sent_file.getValue();
      myParam.encoder_model_file = encoder_model_file.getValue();
      myParam.encoder_reverse_model_file = encoder_reverse_model_file.getValue();
      myParam.decoder_model_file = decoder_model_file.getValue();
      myParam.testing_sent_file = testing_sent_file.getValue();
      myParam.norm_clipping = norm_clipping.getValue();
      myParam.norm_threshold = norm_threshold.getValue();
      myParam.dropout_probability = dropout_probability.getValue();
      myParam.fixed_partition_function = fixed_partition_function.getValue();
      myParam.gradient_check = 0;
      myParam.adagrad_epsilon = 0;
      myParam.use_momentum = 0;
      myParam.init_range = init_range.getValue();
      myParam.init_normal = init_normal.getValue();
      myParam.init_forget = init_forget.getValue();
      myParam.parameter_update = "SGD";
      arg_predicted_sequence_file = predicted_sequence_file.getValue();
      arg_greedy = greedy.getValue();
      arg_stochastic = stochastic.getValue();
      arg_score = score.getValue();
      arg_run_lm = run_lm.getValue();
      arg_carry_states = carry_states.getValue();
      arg_hidden_states_file = hidden_states_file.getValue();
      arg_beam_size = beam_size.getValue();
      do_beam_search = (arg_beam_size > 0);
      arg_reverse_input = reverse_input.getValue();
      arg_min_output_ratio = min_output_ratio.getValue();
      arg_max_output_ratio = max_output_ratio.getValue();
      arg_num_similar_samples = num_similar_samples.getValue();
      arg_similarity_threshold = similarity_threshold.getValue();

	  if (arg_run_lm == 0 && arg_carry_states == 1){
		  cerr<<"--carry_states 1 can only be used with --run_lm 1"<<endl;
		  exit(1);
	  }	  
	  if (arg_hidden_states_file != "")
		  generate_hidden_states = 1;
	  if (arg_greedy + arg_stochastic + arg_score + generate_hidden_states + do_beam_search == 0 || arg_greedy + arg_stochastic + arg_score + do_beam_search >= 2 ) {
		  cerr<<"You have to choose only one option between greedy, stochastic, score, generating hidden states, or do beam search"<<endl;
		  cerr<<"Currently             : "<<endl;
		  cerr<<"greedy                :"<<greedy.getValue()<<endl;
		  cerr<<"stochastic            :"<<stochastic.getValue()<<endl;
		  cerr<<"score                 :"<<score.getValue()<<endl;
		  cerr<<"hidden states file    :"<<hidden_states_file.getValue()<<endl;
		  cerr<<"beam size             :"<<beam_size.getValue()<<endl;
		  exit(1); 
	  }
	  if (arg_greedy + arg_stochastic + do_beam_search >= 1 && arg_predicted_sequence_file == ""){
		  cerr<<"You have to specify a predicted sequence file!"<<endl;
		  exit(1);
	  }

	  myParam.num_threads = num_threads.getValue();
	  
      myParam.minibatch_size = minibatch_size.getValue();
	  if (do_beam_search) {
		  myParam.minibatch_size = 1;
	  }
	  		
      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");

	  cerr << training_sent_file.getDescription() << sep << training_sent_file.getValue() <<endl;
	  cerr << norm_clipping.getDescription() << sep << norm_clipping.getValue() <<endl;
	  cerr << norm_threshold.getDescription() << sep << norm_threshold.getValue() <<endl;
	  cerr << loss_function.getDescription() << sep << loss_function.getValue() <<endl;
	  cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() <<endl;
	  cerr << num_similar_samples.getDescription() << sep << num_similar_samples.getValue() <<endl;
	  cerr << L2_reg.getDescription() << sep << L2_reg.getValue() <<endl;
	  cerr << learning_rate.getDescription() << sep << learning_rate.getValue() <<endl;
	  cerr << testing_sent_file.getDescription() << sep << testing_sent_file.getValue() <<endl;
	  cerr << encoder_model_file.getDescription() << sep << encoder_model_file.getValue() <<endl;
	  cerr << encoder_reverse_model_file.getDescription() << sep << encoder_reverse_model_file.getValue() <<endl;
	  cerr << decoder_model_file.getDescription() << sep << decoder_model_file.getValue() <<endl;
	  cerr << predicted_sequence_file.getDescription()<< sep << predicted_sequence_file.getValue() << endl;
	  
      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
	  cerr << greedy.getDescription()<< sep << greedy.getValue() << endl;
	  cerr << stochastic.getDescription()<< sep << stochastic.getValue() << endl;
	  cerr << score.getDescription()<< sep << score.getValue() << endl;
	  cerr << run_lm.getDescription()<< sep << run_lm.getValue() << endl;
	  cerr << reverse_input.getDescription()<< sep << reverse_input.getValue() << endl;
	  cerr << hidden_states_file.getDescription() << sep << hidden_states_file.getValue() << endl;
	  cerr << beam_size.getDescription() << sep << beam_size.getValue() <<endl;
	  cerr << min_output_ratio.getDescription() << sep << min_output_ratio.getValue() <<endl;
	  cerr << max_output_ratio.getDescription() << sep << max_output_ratio.getValue() <<endl;
    }

    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

	vector<precision_type> sentence_probabilities; 
    myParam.num_threads = setup_threads(myParam.num_threads);
    int save_threads;

    mt19937 rng(time(NULL));

    /////////////////////////READING IN THE TESTING DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read testing data

    vector<int> training_data_flat;
	vector< vector<int> > testing_input_sent, testing_sequence_cont_sent;
	vector< vector<int> > testing_output_sent , validation_sequence_cont_sent;
	vector< vector<int> > decoder_testing_output_sent, decoder_testing_input_sent;
	
	vector< vector<string> > word_testing_input_sent, word_testing_output_sent;
    vec * training_data_flat_mmap;
    data_size_t testing_data_size, validation_data_size; //num_tokens;

	if (arg_run_lm == 1) {
		cerr<<"Running as a LSTM language model"<<endl;
	} else {
		cerr<<"Running as a LSTM sequence to sequence model"<<endl;
	}	
	//Reading the input and output sent files
	data_size_t total_output_tokens, total_input_tokens;
				
	total_output_tokens = total_input_tokens = 0; 

	//data_size
	//readSentFile(myParam.input_sent_file, testing_input_sent,myParam.minibatch_size, total_input_tokens);
	if (arg_run_lm == 0) { //If you're running in LM mode, you do not need to read in the input
		readEvenSentFile(myParam.testing_sent_file, word_testing_input_sent, total_input_tokens,1,0, arg_reverse_input);
	}
	if (arg_score || generate_hidden_states) {
		readOddSentFile(myParam.testing_sent_file, word_testing_output_sent, total_output_tokens,1,1,0);	
	}
	//readSentFile(myParam.output_sent_file, testing_output_sent,myParam.minibatch_size, total_output_tokens);
	if (arg_run_lm == 1)
		testing_data_size = word_testing_output_sent.size();
	else 
		testing_data_size = word_testing_input_sent.size();

	data_size_t num_batches = (testing_data_size-1)/myParam.minibatch_size + 1;

    cerr<<"Number of input tokens "<<total_input_tokens<<endl;
	cerr<<"Number of output tokens "<<total_output_tokens<<endl;
	
    cerr << "Number of testing instances "<< word_testing_input_sent.size() << endl;
	
	Matrix<int,Dynamic,Dynamic> testing_input_sent_data, testing_input_sent_data_reverse, testing_output_sent_data;
	Matrix<int,Dynamic,Dynamic> decoder_testing_output_sent_data, decoder_testing_input_sent_data;
	Array<int,Dynamic,Dynamic> testing_input_sequence_cont_sent_data, testing_input_sequence_cont_sent_data_reverse;
	Array<int,Dynamic,Dynamic> testing_output_sequence_cont_sent_data;	
    
    ///// Create and initialize the neural network and associated propagators.
	vector<string> encoder_input_words, encoder_output_words;
	vector<string> decoder_input_words, decoder_output_words;
	
	model encoder_nn,encoder_nn_reverse,decoder_nn;
	google_input_model encoder_input, decoder_input;
	hidden_to_hidden_input_model feed_input;
	vocabulary encoder_vocab, decoder_vocab, decoder_input_vocab, decoder_output_vocab;
	if (arg_run_lm == 0) {
		encoder_nn.read(myParam.encoder_model_file, encoder_input_words, encoder_output_words);
		encoder_nn_reverse.read(myParam.encoder_reverse_model_file, encoder_input_words, encoder_output_words);
		encoder_input.read(myParam.encoder_model_file);
		encoder_vocab.build_vocab(encoder_input_words);
		//initialize gradient
		encoder_nn.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			-log(encoder_vocab.size()),
			myParam.init_forget,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
		encoder_nn_reverse.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			-log(encoder_vocab.size()),
			myParam.init_forget,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
		encoder_input.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
		encoder_nn.set_input(encoder_input);
		encoder_nn_reverse.set_input(encoder_input);
		//encoder_nn.write(myParam.encoder_model_file+".debug", encoder_input_words, encoder_output_words);
		//encoder_nn_reverse.write(myParam.encoder_reverse_model_file+".debug", encoder_input_words, encoder_output_words);
	} else {
		//this is cheating for now
		encoder_nn.read(myParam.decoder_model_file, encoder_input_words, encoder_output_words);
		encoder_nn_reverse.read(myParam.decoder_model_file, encoder_input_words, encoder_output_words);
		encoder_input.read(myParam.decoder_model_file);
		encoder_nn.set_input(encoder_input);
		encoder_nn_reverse.set_input(encoder_input);
		encoder_vocab.build_vocab(encoder_input_words);		
	}
	
	decoder_nn.read(myParam.decoder_model_file, decoder_input_words, decoder_output_words);
	decoder_input_vocab.build_vocab(decoder_input_words);
	decoder_output_vocab.build_vocab(decoder_output_words);
	decoder_nn.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			-log(decoder_output_vocab.size()),
			myParam.init_forget,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
	
	arg_output_start_symbol = decoder_input_vocab.lookup_word("<s>");
	arg_output_end_symbol = decoder_output_vocab.lookup_word("</s>");
	
	if (arg_beam_size > decoder_output_vocab.size()) {
		cerr<<"Warning: The beam size is larger than the output vocabulary."<<endl;
		cerr<<"User specified beam size: "<<arg_beam_size<<endl;
		cerr<<"Output vocabulary size: "<<decoder_output_vocab.size()<<endl;
	}

	decoder_input.read(myParam.decoder_model_file);
	decoder_input.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
	
	decoder_nn.set_input(decoder_input);

	if( decoder_nn.feed_input_flag ){
		feed_input.read(myParam.decoder_model_file);
		feed_input.initializeGradient(rng,
			myParam.init_normal,
			myParam.init_range,
			myParam.parameter_update,
			myParam.adagrad_epsilon);
		decoder_nn.set_feed_input(feed_input);
	}

	//decoder_nn.write(myParam.decoder_model_file+".debug", decoder_input_words, decoder_output_words);
	
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL
	myParam.num_hidden = encoder_nn.get_hidden();
	myParam.input_embedding_dimension = myParam.num_hidden;
	myParam.output_embedding_dimension = myParam.num_hidden;
	cerr<<"done reading the models "<<endl;
	
	//Transforming the input and output data
	if (arg_run_lm == 0) {
		integerize(word_testing_input_sent, 
						testing_input_sent, 
						encoder_vocab);
	}
	if (arg_score || generate_hidden_states) { 
		integerize(word_testing_output_sent, 
						testing_output_sent, 
						decoder_vocab);	
		integerize(word_testing_output_sent, 
				   decoder_testing_output_sent,
				   decoder_output_vocab,
				   1,
				   0);
	   	integerize(word_testing_output_sent, 
	   			   decoder_testing_input_sent,
	   			   decoder_input_vocab,
	   			   0,
	   			   1);								
	}

	propagator<Google_input_node, Hidden_to_hidden_input_node, google_input_model, hidden_to_hidden_input_model> prop(encoder_nn, encoder_nn_reverse, decoder_nn, myParam.minibatch_size);

	/////////////////////////READING IN THE TRAINING DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////
	
	vector< vector<int> > training_input_sent, training_sequence_cont_sent;
	vector< vector<int> > training_output_sent; 
	vector< vector<int> > decoder_training_input_sent, decoder_training_output_sent;
	vector< vector<string> > word_training_input_sent, word_training_output_sent;

	data_size_t training_data_size; 
	
	//Reading the input and output sent files
	data_size_t total_training_output_tokens,
				total_training_input_tokens;
				
	total_training_output_tokens = 
		total_training_input_tokens = 0;

	//Even sentences are input
	if( arg_run_lm == 0 ) {
		readEvenSentFile(myParam.training_sent_file, word_training_input_sent, total_training_input_tokens, 1, 0, myParam.reverse_input); //1: add sentence start and end symbol; 0: indicates input source sentences
		integerize(word_training_input_sent, 
						training_input_sent, 
						encoder_vocab); //transform the word string to word id
    	cerr<<"Number of input tokens "<<total_training_input_tokens<<endl;						
	}

	//Reading output 
	readOddSentFile(myParam.training_sent_file, word_training_output_sent, total_training_output_tokens,1,1, 0); //We do not reverse the output
				
	integerize(word_training_output_sent, 
			   decoder_training_output_sent,
			   decoder_output_vocab,
			   1,
			   0); //preparing decoder output: from 1 to n of a sentence
   	integerize(word_training_output_sent, 
   			   decoder_training_input_sent,
   			   decoder_input_vocab,
   			   0,
   			   1); //preparing decoder input: from 0 to n-1 of a sentence
	assert(decoder_training_output_sent.size() == decoder_training_input_sent.size());   						
	
	training_data_size = decoder_training_output_sent.size();

	cerr<<"Number of output tokens "<<total_training_output_tokens<<endl;
	cerr << "Number of training instances "<< decoder_training_output_sent.size() << endl;


	/*
	Matrix<precision_type,Dynamic,Dynamic,RowMajor> &input_word_embeddings = (dynamic_cast<google_input_model&>(*(encoder_nn.input))).input_layer.word_embeddings();
	Matrix<precision_type,Dynamic,Dynamic> training_input_sent_embeddings, testing_input_sent_embeddings;
	training_input_sent_embeddings.setZero(training_data_size, input_word_embeddings.cols());
	testing_input_sent_embeddings.setZero(testing_data_size, input_word_embeddings.cols());
	sentToEmbedding(training_input_sent, training_input_sent_embeddings, input_word_embeddings);
	sentToEmbedding(testing_input_sent, testing_input_sent_embeddings, input_word_embeddings);
	*/
	vector< vector<int> > similar_sample_vector(testing_data_size, vector<int>());
	//findSimilarSamples(training_input_sent_embeddings, testing_input_sent_embeddings, similar_sample_vector, arg_num_similar_samples);
	findSimilarSamples(word_training_input_sent, word_testing_input_sent, similar_sample_vector, arg_num_similar_samples, arg_similarity_threshold);
	/*
	for( int i = 0; i < testing_data_size; ++i )
	{
		cerr<<"similar samples:"<<endl;
		for( int j = 0; j < word_testing_input_sent[i].size(); ++j )
		{
			cerr<<word_testing_input_sent[i][j]<<" ";
		}
		cerr<<endl;
		for( int j = 0; j < similar_sample_vector[i].size(); ++j )
		{
			for( int k = 0; k < word_training_input_sent[similar_sample_vector[i][j]].size(); ++k )
				cerr<<word_training_input_sent[similar_sample_vector[i][j]][k]<<" ";
			cerr<<endl;
		}
		cerr<<endl;
	}
	*/

	loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

	multinomial<data_size_t> unigram;																
	if (loss_function == NCELoss){
		vector<data_size_t> unigram_counts = vector<data_size_t>(decoder_output_vocab.size(),0);
		for (int sent_index=0; sent_index<decoder_training_output_sent.size(); sent_index++){
			for (int word_index=0; word_index<decoder_training_output_sent[sent_index].size(); word_index++){
				unigram_counts[decoder_training_output_sent[sent_index][word_index]] += 1;
			}
		}
		unigram = multinomial<data_size_t> (unigram_counts);
	}
	if (loss_function == NCELoss)
		prop.resizeNCE(myParam.num_noise_samples, myParam.fixed_partition_function, unigram);

    ///////////////////////TESTING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
	
	//Resetting the gradient in the beginning
	prop.resetGradient();

	cerr << "Testing minibatches: ";

	precision_type log_likelihood = 0.0;
	
	precision_type data_log_likelihood=0;	
	Matrix<precision_type,Dynamic,Dynamic> current_c_for_gradCheck, current_h_for_gradCheck, current_c,current_h, init_c, init_h, source_ctx_mean, source_cell_mean;
	current_c.setZero(myParam.num_hidden, minibatch_size);
	current_h.setZero(myParam.num_hidden, minibatch_size);

	ofstream file;
	if (arg_predicted_sequence_file != "") {
    	file.open(arg_predicted_sequence_file.c_str(), std::ofstream::out ); //we are not appending| std::ofstream::app);
    	if (!file) throw runtime_error("Could not open file " + arg_predicted_sequence_file);
	}

	//precision_type log_likelihood = 0.0;
	ofstream hidden_states_file;
	if (arg_hidden_states_file != ""){
		hidden_states_file.open(arg_hidden_states_file.c_str(),std::ofstream::out )	;
	} 
	
	vector< Matrix<precision_type,Dynamic,Dynamic> > source_ctx(105, Matrix<precision_type,Dynamic,Dynamic>());
	vector< Matrix<precision_type,Dynamic,Dynamic> > source_cell(105, Matrix<precision_type,Dynamic,Dynamic>());
	for( int src_id = 0; src_id < 105; src_id++ ) {
		source_ctx[src_id].setZero(myParam.num_hidden*2, minibatch_size);
		source_cell[src_id].setZero(myParam.num_hidden*2, minibatch_size);
	}

	for(data_size_t batch=0;batch<num_batches;batch++)
	{

		//One Sentence One Model for Neural Machine Translation
		//ADAPT TRAINING THE NEURAL NETWORK WITH SIMILAR TRAINING DATA
		/***********************************************************************************
		**********************************************************************************/

		if( similar_sample_vector[batch].size() )
		{
			int cur_num_similar_samples = similar_sample_vector[batch].size();
			precision_type current_momentum = -1;
			precision_type current_learning_rate = myParam.learning_rate;

			int num_noise_samples = myParam.num_noise_samples;

			//Resetting the gradient in the beginning
			prop.resetGradient();

			//initialize the source context representation
			vector< Matrix<precision_type,Dynamic,Dynamic> > training_source_ctx(105, Matrix<precision_type,Dynamic,Dynamic>()); //only allow for max 105 input words
			vector< Matrix<precision_type,Dynamic,Dynamic> > training_source_cell(105, Matrix<precision_type,Dynamic,Dynamic>());
			for( int src_id = 0; src_id < 105; src_id++ ) {
				training_source_ctx[src_id].setZero(myParam.num_hidden*2, minibatch_size);
				training_source_cell[src_id].setZero(myParam.num_hidden*2, minibatch_size);
			}

			int max_epochs = 1;
			for( int epoch = 0; epoch < max_epochs; ++epoch )
			{ 

				cerr << "Training minibatches: ";

				precision_type log_likelihood = 0.0;

				int num_samples = 0;
				if( loss_function == LogLoss )
					num_samples = output_vocab_size;
				else if( loss_function == NCELoss )
					num_samples = 1+num_noise_samples;

				data_size_t training_num_batches = (cur_num_similar_samples-1)/myParam.minibatch_size + 1;
				precision_type training_data_log_likelihood=0;	
				//initialize some variables
				Matrix<precision_type,Dynamic,Dynamic> training_current_c, training_current_h, training_init_c, training_init_h, training_init_ctx_mean, training_init_cell_mean;
				training_init_c.setZero(myParam.num_hidden,minibatch_size);
				training_init_h.setZero(myParam.num_hidden,minibatch_size);
				training_current_c.setZero(myParam.num_hidden, minibatch_size);
				training_current_h.setZero(myParam.num_hidden, minibatch_size);			
				training_init_ctx_mean.setZero(myParam.num_hidden*2, minibatch_size);
				training_init_cell_mean.setZero(myParam.num_hidden*2, minibatch_size);			

				for( data_size_t training_batch=0; training_batch < training_num_batches; training_batch++ )
				{
					if (arg_carry_states == 0) {
						training_current_c.setZero(myParam.num_hidden, minibatch_size);
						training_current_h.setZero(myParam.num_hidden, minibatch_size);			
					}
					if (training_batch > 0 && training_batch % 100 == 0)
					{
						cerr << training_batch <<"...";
					} 

					//determine minibatch start index and end index
					data_size_t training_minibatch_start_index = minibatch_size * training_batch;
					data_size_t training_minibatch_end_index = min(static_cast<data_size_t>(cur_num_similar_samples-1), static_cast<data_size_t> (training_minibatch_start_index+minibatch_size-1));
					int training_current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), static_cast<data_size_t>(cur_num_similar_samples) - training_minibatch_start_index);

					//Getting a minibatch of sentences
					vector<int> training_minibatch_input_sentences, training_minibatch_input_sentences_reverse, training_minibatch_input_token_num;
					vector<int> training_minibatch_input_sequence_cont_sentences, training_minibatch_input_sequence_cont_sentences_reverse, training_minibatch_output_sequence_cont_sentences;
					vector<int> training_minibatch_decoder_output_sentences, training_minibatch_decoder_input_sentences;
					unsigned int training_max_input_sent_len, training_max_output_sent_len;
					unsigned int training_minibatch_output_tokens, training_minibatch_input_tokens, training_minibatch_sequence_cont_tokens;
					training_minibatch_output_tokens = training_minibatch_input_tokens = training_minibatch_sequence_cont_tokens = 0;
					training_max_input_sent_len = training_max_output_sent_len = 0;

					training_init_c = training_current_c;
					training_init_h = training_current_h; 	

					if (arg_run_lm == 0) {
						//select the training_input_sent from minibatch_start_index to minibatch_end_index to form minibatch_input_sentences and minibatch_input_sentences_reverse; minibatch_input_token_num stores the token number for each sentence in the batch; max_input_sent_len keeps the sentence max_length in the minibatch; minibatch_input_tokens stores the total token number of the minibatch; 1 stands for data (0 stands for continuation flags)
						/*
						for( int k = 0; k < training_input_sent[similar_sample_vector[batch][training_minibatch_start_index]].size(); ++k )
							cerr<<training_input_sent[similar_sample_vector[batch][training_minibatch_start_index]][k]<<" ";
						cerr<<endl;
						cerr<<endl;
						*/

						miniBatchifyEncoder(training_input_sent, 
								similar_sample_vector[batch],
								training_minibatch_input_sentences,
								training_minibatch_input_sentences_reverse,
								training_minibatch_input_token_num,
								training_minibatch_start_index,
								training_minibatch_end_index,
								training_max_input_sent_len,
								training_minibatch_input_tokens,
								1);	
						training_minibatch_input_tokens = 0;
						miniBatchifyEncoder(training_input_sent, 
								similar_sample_vector[batch],
								training_minibatch_input_sequence_cont_sentences,
								training_minibatch_input_sequence_cont_sentences_reverse,
								training_minibatch_input_token_num,
								training_minibatch_start_index,
								training_minibatch_end_index,
								training_max_input_sent_len,
								training_minibatch_input_tokens,
								0);			
						/*
						cerr<<"after batchify ..."<<endl;
						for( int k = 0; k < training_minibatch_input_sentences.size(); ++k )
							cerr<<training_minibatch_input_sentences[k]<<" ";
						cerr<<endl;
						cerr<<endl;

						for( int k = 0; k < training_minibatch_input_sentences_reverse.size(); ++k )
							cerr<<training_minibatch_input_sentences_reverse[k]<<" ";
						cerr<<endl;
						cerr<<endl;
						*/
						//convert the minibatch data and continuation vectors into matrix
						Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data (training_minibatch_input_sentences.data(), 
								training_max_input_sent_len,
								training_current_minibatch_size);
						Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data (training_minibatch_input_sequence_cont_sentences.data(),
								training_max_input_sent_len,
								training_current_minibatch_size);	
						Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data_reverse (training_minibatch_input_sentences_reverse.data(), 
								training_max_input_sent_len,
								training_current_minibatch_size);
						Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data_reverse (training_minibatch_input_sequence_cont_sentences_reverse.data(),
								training_max_input_sent_len,
								training_current_minibatch_size);	
						if (myParam.dropout_probability > 0.) {
							//encode the source sentences with left-to-right LSTM
							prop.fPropEncoderDropout(training_input_sent_data,
									training_current_c,
									training_current_h,
									training_input_sequence_cont_sent_data,
									rng);					
							training_current_c = training_init_c;
							training_current_h = training_init_h;
							//encode the source sentences with right-to-left LSTM
							prop.fPropEncoderDropoutReverse(training_input_sent_data_reverse,
									training_current_c,
									training_current_h,
									training_input_sequence_cont_sent_data_reverse,
									rng);					
						} else {
							prop.fPropEncoder(training_input_sent_data,
									training_current_c,
									training_current_h,
									training_input_sequence_cont_sent_data);
							training_current_c = training_init_c;
							training_current_h = training_init_h;
							prop.fPropEncoderReverse(training_input_sent_data_reverse,
									training_current_c,
									training_current_h,
									training_input_sequence_cont_sent_data_reverse);
						}
					}

					//concatenate the forward and backward source contexts
					#pragma omp parallel for
					for( int i = 0; i < training_max_input_sent_len; ++i ){
						int j = 0;
						for( j = 0; j < training_current_minibatch_size; ++j ){
							//minibatch_input_token_num[j] stores the actural token number for the jth sentence in the minibatch
							//set the padded words in the beginning and initialize them zeros
							int padded_word_num = training_max_input_sent_len - training_minibatch_input_token_num[j];
							if( i < padded_word_num ) {
								training_source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
									prop.encoder_lstm_nodes_reverse[i].h_t.col(j);
								training_source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
									prop.encoder_lstm_nodes_reverse[i].c_t.col(j);
							}
							else {
								int reverse_id = training_max_input_sent_len-i+padded_word_num-1; //reverse_id in right_to_left encoder points at the same word as i in the left_to_right encoder 
								training_source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
									prop.encoder_lstm_nodes_reverse[reverse_id].h_t.col(j);
								training_source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
									prop.encoder_lstm_nodes_reverse[reverse_id].c_t.col(j);
							}
						}
						for( ; j < minibatch_size; ++j ) {
							training_source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
								prop.encoder_lstm_nodes_reverse[i].h_t.col(j);
							training_source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
								prop.encoder_lstm_nodes_reverse[i].c_t.col(j);
						}
					}

					//use the average context as the initial context of the decoder
					Matrix<precision_type,Dynamic,Dynamic> training_source_ctx_mean = training_init_ctx_mean;
					Matrix<precision_type,Dynamic,Dynamic> training_source_cell_mean = training_init_cell_mean;
					for( int i = 0; i < training_max_input_sent_len; ++i ){
						training_source_ctx_mean += training_source_ctx[i];
						training_source_cell_mean += training_source_cell[i];
					}
					#pragma omp parallel for
					for( int i = 0; i < training_current_minibatch_size; ++i ){
						training_source_ctx_mean.col(i) = training_source_ctx_mean.col(i)/training_minibatch_input_token_num[i];
						training_source_cell_mean.col(i) = training_source_cell_mean.col(i)/training_minibatch_input_token_num[i];
					}


					//preparing the data info for the decoder
					//extract the decoder_training_output_sent from minibatch_start_index to minibatch_end_index to minibatch_decoder_output_sentences; max_output_sent_len keeps the sentence max length in the minibatch; minibatch_output_tokens stores the total token number in the minibatch; 1 stands for preparing data; -1 indicates the padded value
					/*
					cerr<<"before batchify decoder ..."<<endl;
					for( int k = 0; k < decoder_training_input_sent[similar_sample_vector[batch][training_minibatch_start_index]].size(); ++k )
						cerr<<decoder_training_input_sent[similar_sample_vector[batch][training_minibatch_start_index]][k]<<" ";
					cerr<<endl;
					cerr<<endl;

					for( int k = 0; k < decoder_training_output_sent[similar_sample_vector[batch][training_minibatch_start_index]].size(); ++k )
						cerr<<decoder_training_output_sent[similar_sample_vector[batch][training_minibatch_start_index]][k]<<" ";
					cerr<<endl;
					cerr<<endl;
					*/

					miniBatchifyDecoder(decoder_training_output_sent, 
							similar_sample_vector[batch],
							training_minibatch_decoder_output_sentences,
							training_minibatch_start_index,
							training_minibatch_end_index,
							training_max_output_sent_len,
							training_minibatch_output_tokens,
							1,
							-1);
					training_minibatch_output_tokens =0;						
					miniBatchifyDecoder(decoder_training_input_sent, 
							similar_sample_vector[batch],
							training_minibatch_decoder_input_sentences,
							training_minibatch_start_index,
							training_minibatch_end_index,
							training_max_output_sent_len,
							training_minibatch_output_tokens,
							1,
							0);									
					/*
					cerr<<"after batchify for decoder ..."<<endl;
					for( int k = 0; k < training_minibatch_decoder_input_sentences.size(); ++k )
						cerr<<training_minibatch_decoder_input_sentences[k]<<" ";
					cerr<<endl;
					cerr<<endl;

					for( int k = 0; k < training_minibatch_decoder_output_sentences.size(); ++k )
						cerr<<training_minibatch_decoder_output_sentences[k]<<" ";
					cerr<<endl;
					cerr<<endl;
					*/

					//convert vectors to maxtrix
					Map< Matrix<int, Dynamic, Dynamic> >  decoder_training_output_sent_data (training_minibatch_decoder_output_sentences.data(),
							training_max_output_sent_len,
							training_current_minibatch_size);
					Map< Matrix<int, Dynamic, Dynamic> >  decoder_training_input_sent_data (training_minibatch_decoder_input_sentences.data(),
							training_max_output_sent_len,
							training_current_minibatch_size);	

					//preparing the continuation vectors and convert them to matrix
					training_minibatch_output_tokens =0;	
					miniBatchifyDecoder(decoder_training_input_sent, 
							similar_sample_vector[batch],
							training_minibatch_output_sequence_cont_sentences,
							training_minibatch_start_index,
							training_minibatch_end_index,
							training_max_output_sent_len,
							training_minibatch_output_tokens,
							0,
							0);	
					Map< Array<int,Dynamic,Dynamic> > training_output_sequence_cont_sent_data(training_minibatch_output_sequence_cont_sentences.data(),
							training_max_output_sent_len,
							training_current_minibatch_size);

					if (myParam.dropout_probability > 0.) {
						//decode the minibatch sentences with LSTM: decoder_training_input_sent_data is the data input of LSTM; source_cell_mean and source_ctx_mean serve as the initial history of the decoder LSTM; source_ctx is used for attention model; minibatch_input_token_num stores the actural token number for each sentence in the minibatch
						prop.fPropDecoderDropout(decoder_training_input_sent_data,
								training_source_cell_mean,
								training_source_ctx_mean,
								training_source_ctx,
								training_output_sequence_cont_sent_data,
								training_minibatch_input_token_num,
								training_max_input_sent_len,
								rng);				
					} else {
						prop.fPropDecoder(decoder_training_input_sent_data,
								training_source_cell_mean,
								training_source_ctx_mean,
								training_source_ctx,
								training_output_sequence_cont_sent_data,
								training_minibatch_input_token_num,
								training_max_input_sent_len);
					}

					precision_type adjusted_learning_rate = current_learning_rate;
					if (!myParam.norm_clipping){
						adjusted_learning_rate /= training_current_minibatch_size;			
					}

					//computing losses
					if (myParam.dropout_probability > 0) {
						//compute the losses according to the decoder output
						prop.computeLossesDropout(decoder_training_output_sent_data,
								training_data_log_likelihood,
								myParam.gradient_check,
								myParam.norm_clipping,
								loss_function,
								unigram,
								num_noise_samples,
								rng);
						//do decoder LSTM backpropagation after obtaining loss
						prop.bPropDecoderDropout(decoder_training_input_sent_data,
								training_source_ctx,
								training_minibatch_input_token_num,
								training_max_input_sent_len,
								myParam.gradient_check,
								myParam.norm_clipping); 
					} else {
						prop.computeLosses(decoder_training_output_sent_data,
								training_data_log_likelihood,
								myParam.gradient_check,
								myParam.norm_clipping,
								loss_function,
								unigram,
								num_noise_samples,
								rng);
						prop.bPropDecoder(decoder_training_input_sent_data,
								training_source_ctx,
								training_minibatch_input_token_num,
								training_max_input_sent_len,
								myParam.gradient_check,						  
								myParam.norm_clipping);
					}	


					if (arg_run_lm == 0) { 
						Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data (training_minibatch_input_sentences.data(), 
								training_max_input_sent_len,
								training_current_minibatch_size);
						Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data (training_minibatch_input_sequence_cont_sentences.data(),
								training_max_input_sent_len,
								training_current_minibatch_size);						
						Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data_reverse (training_minibatch_input_sentences_reverse.data(), 
								training_max_input_sent_len,
								training_current_minibatch_size);
						Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data_reverse (training_minibatch_input_sequence_cont_sentences_reverse.data(),
								training_max_input_sent_len,
								training_current_minibatch_size);						
						if (myParam.dropout_probability > 0.){ 
							//left_to_right and right_to_left encoder LSTM backpropagation
							prop.bPropEncoderDropout(training_input_sent_data,
									myParam.gradient_check,
									myParam.norm_clipping,
									training_input_sequence_cont_sent_data); 	
							prop.bPropEncoderDropoutReverse(training_input_sent_data_reverse,
									myParam.gradient_check,
									myParam.norm_clipping,
									training_input_sequence_cont_sent_data_reverse); 	
						} else {
							prop.bPropEncoder(training_input_sent_data,
									myParam.gradient_check,
									myParam.norm_clipping,
									training_input_sequence_cont_sent_data);
							prop.bPropEncoderReverse(training_input_sent_data_reverse,
									myParam.gradient_check,
									myParam.norm_clipping,
									training_input_sequence_cont_sent_data_reverse);
						}				 
					}

					//Updaging using local grad norms and SGD
					prop.updateParams(adjusted_learning_rate,
							training_max_input_sent_len + training_max_output_sent_len,
							current_momentum,
							myParam.L2_reg,
							myParam.norm_clipping,
							myParam.norm_threshold,
							loss_function,
							arg_run_lm);	

					//Resetting the gradients
					prop.resetGradient();
				}
			}

		}

		/***********************************************************************************
		**********************************************************************************/
		if (arg_carry_states == 0) {
			current_c.setZero(myParam.num_hidden, minibatch_size);
			current_h.setZero(myParam.num_hidden, minibatch_size);		
		}
		precision_type minibatch_log_likelihood = 0.;
		if (batch > 0 && batch % 100 == 0)
		{
			cerr << batch <<"...";
		} 

		data_size_t minibatch_start_index = minibatch_size * batch;
		data_size_t minibatch_end_index = min(testing_data_size-1, static_cast<data_size_t> (minibatch_start_index+minibatch_size-1));

		int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), testing_data_size - minibatch_start_index);
		vector<vector<int> > predicted_sequence(current_minibatch_size);

		///// Forward propagation
		//Taking the input and output sentence and setting the testing data to it.
		//Getting a minibatch of sentences
		vector<int> minibatch_input_sentences, 
			minibatch_input_sentences_reverse,
			minibatch_output_sentences, 
			minibatch_decoder_output_sentences,
			minibatch_decoder_input_sentences,
			minibatch_input_sequence_cont_sentences,
			minibatch_input_sequence_cont_sentences_reverse,
			minibatch_input_token_num,
			minibatch_output_sequence_cont_sentences;
		unsigned int max_input_sent_len, max_output_sent_len;
		max_input_sent_len = max_output_sent_len = 0;
		unsigned int minibatch_output_tokens,minibatch_input_tokens, minibatch_sequence_cont_tokens;
		minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;
		if (arg_run_lm == 0) {
			miniBatchifyEncoder(testing_input_sent, 
					minibatch_input_sentences,
					minibatch_input_sentences_reverse,
					minibatch_input_token_num,
					minibatch_start_index,
					minibatch_end_index,
					max_input_sent_len,
					minibatch_input_tokens,
					1);	
			minibatch_input_tokens = 0;
			miniBatchifyEncoder(testing_input_sent, 
					minibatch_input_sequence_cont_sentences,
					minibatch_input_sequence_cont_sentences_reverse,
					minibatch_input_token_num,
					minibatch_start_index,
					minibatch_end_index,
					max_input_sent_len,
					minibatch_input_tokens,
					0);		

			testing_input_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_input_sentences.data(), 
					max_input_sent_len,
					current_minibatch_size);
			testing_input_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_input_sequence_cont_sentences.data(),
					max_input_sent_len,
					current_minibatch_size);
			testing_input_sent_data_reverse = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_input_sentences_reverse.data(), 
					max_input_sent_len,
					current_minibatch_size);
			testing_input_sequence_cont_sent_data_reverse = Map< Array<int,Dynamic,Dynamic> >(minibatch_input_sequence_cont_sentences_reverse.data(),
					max_input_sent_len,
					current_minibatch_size);
		}																
		if (arg_score == 1 || generate_hidden_states == 1) {
			miniBatchifyDecoder(decoder_testing_output_sent, 
					minibatch_decoder_output_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					1,
					-1);		
			minibatch_output_tokens =0;						
			miniBatchifyDecoder(decoder_testing_input_sent, 
					minibatch_decoder_input_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					1,
					0);									
			decoder_testing_output_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_output_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);
			decoder_testing_input_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_input_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);	

			minibatch_output_tokens =0;
			miniBatchifyDecoder(decoder_testing_input_sent, 
					minibatch_output_sequence_cont_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					0);		
			testing_output_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_output_sequence_cont_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);

		}																			

		init_c = current_c;
		init_h = current_h; 	
		vector<Matrix<precision_type, Dynamic, Dynamic> > input_hiddens,
			encoder_h_t,
			encoder_o_t,
			encoder_c_t,
			encoder_i_t,
			encoder_f_t,
			decoder_h_t,
			decoder_o_t,
			decoder_c_t,
			decoder_i_t,
			decoder_f_t;

		vector<Matrix<precision_type, Dynamic, Dynamic> > output_hiddens;
		if (arg_run_lm == 0) {		
			prop.fPropEncoder(testing_input_sent_data,
					current_c,
					current_h,
					testing_input_sequence_cont_sent_data);	
			current_c = init_c;
			current_h = init_h;
			prop.fPropEncoderReverse(testing_input_sent_data_reverse,
					current_c,
					current_h,
					testing_input_sequence_cont_sent_data_reverse);

			//printing out the encoder states along with the sentence
			if (arg_hidden_states_file != ""){
				for(int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++) {
					Matrix< precision_type, Dynamic, Dynamic> hidden_states,
						internal_h_t,
						internal_c_t,
						internal_i_t,
						internal_f_t,
						internal_o_t; 
					hidden_states.setZero(myParam.num_hidden,max_input_sent_len);
					internal_h_t.setZero(myParam.num_hidden,max_input_sent_len);
					internal_c_t.setZero(myParam.num_hidden,max_input_sent_len);
					internal_i_t.setZero(myParam.num_hidden,max_input_sent_len);
					internal_f_t.setZero(myParam.num_hidden,max_input_sent_len);
					internal_o_t.setZero(myParam.num_hidden,max_input_sent_len);


					//Getting all the hidden states for the particular sentence
					prop.getHiddenStates(hidden_states,
							max_input_sent_len,
							1,
							minibatch_index); 
					prop.getInternals(internal_h_t,
							internal_c_t,
							internal_f_t,
							internal_i_t,
							internal_o_t,
							max_input_sent_len,
							1,
							minibatch_index);										
					//prob.getInternals	
					input_hiddens.push_back(hidden_states);
					encoder_h_t.push_back(internal_h_t);
					encoder_o_t.push_back(internal_o_t);
					encoder_f_t.push_back(internal_f_t);
					encoder_i_t.push_back(internal_i_t);
					encoder_c_t.push_back(internal_c_t);
				}													
			}
		}

		//concatenate the forward and backward source contexts
		#pragma omp parallel for
		for( int i = 0; i < max_input_sent_len; ++i ){
			for( int j = 0; j < current_minibatch_size; ++j ){
				int padded_word_num = max_input_sent_len - minibatch_input_token_num[j];
				if( i < padded_word_num ) {
					source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
						prop.encoder_lstm_nodes_reverse[i].h_t.col(j);
					source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
						prop.encoder_lstm_nodes_reverse[i].c_t.col(j);
				}
				else {
					source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
						prop.encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].h_t.col(j);
					source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
						prop.encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].c_t.col(j);
				}
			}
		}

		source_ctx_mean.setZero(myParam.num_hidden*2, minibatch_size);
		source_cell_mean.setZero(myParam.num_hidden*2, minibatch_size);
		for( int i = 0; i < max_input_sent_len; ++i ){
			source_ctx_mean += source_ctx[i];
			source_cell_mean += source_cell[i];
		}

		#pragma omp parallel for
		for( int i = 0; i < current_minibatch_size; ++i ){
			source_ctx_mean.col(i) = source_ctx_mean.col(i)/minibatch_input_token_num[i];
			source_cell_mean.col(i) = source_cell_mean.col(i)/minibatch_input_token_num[i];
		}

		if (arg_greedy) {
			//cerr<<"Generating greedy output"<<endl;
			prop.generateGreedyOutput(testing_input_sent_data,
					decoder_input_vocab,
					decoder_output_vocab,
					source_cell_mean,
					source_ctx_mean,		
					source_ctx,
					minibatch_input_token_num,
					predicted_sequence,
					arg_output_start_symbol,
					arg_output_end_symbol);				
		}
		if (arg_stochastic){
			prop.generateStochasticOutput(testing_input_sent_data,
					decoder_input_vocab,
					decoder_output_vocab,				
					source_cell_mean,
					source_ctx_mean,		
					source_ctx,
					minibatch_input_token_num,
					predicted_sequence,
					arg_output_start_symbol,
					arg_output_end_symbol,
					rng);
		}
		if (do_beam_search) {
			//Need to resize the propagator to the beam size
			prop.resize(arg_beam_size);
			vector<k_best_seq_item> final_k_best_seq_list;
			prop.beamDecoding(testing_input_sent_data,
					decoder_input_vocab,
					decoder_output_vocab,
					source_cell_mean,
					source_ctx_mean,
					source_ctx,
					minibatch_input_token_num,
					final_k_best_seq_list,
					arg_output_start_symbol,
					arg_output_end_symbol,
					arg_beam_size);
			//Now printing the k best list
			file<<"Source\t:";
			//ASSUMING THAT THE MINIBATCH SIZE IS 1 FOR NOW!! HAVE TO CHANGE THIS!!
			//Starting with word index 1 because we don't want to print <s>
			for (int word_index=1; word_index<max_input_sent_len; word_index++){																
				file << encoder_vocab.get_word(testing_input_sent_data(word_index,0))<<" ";
			}
			file<<endl;

			if (final_k_best_seq_list.size() < arg_beam_size) {
				cerr<<"The n-best list for sentence has "<<final_k_best_seq_list.size()<<" items"<<endl;
			}

			int output_seq_num = 0;
			for (int sent_id=0; sent_id<final_k_best_seq_list.size(); sent_id++){	
				if( final_k_best_seq_list.at(sent_id).seq.size() < arg_min_output_ratio*max_input_sent_len )
					continue;

				if( final_k_best_seq_list.at(sent_id).seq.size() > arg_max_output_ratio*max_input_sent_len )
					continue;

				file<<"Hyp"<<sent_id<<"\t\t:";
				for(int word_id=0; word_id<final_k_best_seq_list.at(sent_id).seq.size(); word_id++){
					if (final_k_best_seq_list.at(sent_id).seq.at(word_id) == arg_output_end_symbol) {
						break;
					} else {
						file << decoder_output_vocab.get_word(final_k_best_seq_list.at(sent_id).seq.at(word_id))<<" ";
					}	
				}
				file<<"Sequence probability: "<<exp(final_k_best_seq_list.at(sent_id).value)<<endl;
				output_seq_num += 1;
				if( output_seq_num > arg_beam_size )
					break;
			}
			file<<endl;
		}
		if (arg_score || generate_hidden_states ) {
			prop.fPropDecoder(decoder_testing_input_sent_data,
					source_cell_mean,
					source_ctx_mean,
					source_ctx,
					testing_output_sequence_cont_sent_data,
					minibatch_input_token_num,
					max_input_sent_len);
			if (generate_hidden_states) {
				//cerr<<"Getting hiddens from decoder"<<endl;
				for(int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++) {

					Matrix< precision_type, Dynamic, Dynamic> hidden_states,
						internal_h_t,
						internal_c_t,
						internal_i_t,
						internal_f_t,
						internal_o_t; 
					internal_h_t.setZero(myParam.num_hidden*2,max_input_sent_len);
					internal_c_t.setZero(myParam.num_hidden*2,max_input_sent_len);
					internal_i_t.setZero(myParam.num_hidden*2,max_input_sent_len);
					internal_f_t.setZero(myParam.num_hidden*2,max_input_sent_len);
					internal_o_t.setZero(myParam.num_hidden*2,max_input_sent_len);						 
					hidden_states.resize(myParam.num_hidden*2,max_output_sent_len-1);
					hidden_states.setZero();						

					prop.getHiddenStates(hidden_states,
							max_output_sent_len-1,
							0,
							minibatch_index); 

					prop.getInternals(internal_h_t,
							internal_c_t,
							internal_f_t,
							internal_i_t,
							internal_o_t,
							max_output_sent_len-1,
							0,
							minibatch_index);	

					output_hiddens.push_back(hidden_states);
					decoder_h_t.push_back(internal_h_t);
					decoder_o_t.push_back(internal_o_t);
					decoder_f_t.push_back(internal_f_t);
					decoder_i_t.push_back(internal_i_t);
					decoder_c_t.push_back(internal_c_t);						
				}
			}							
		}
		if (arg_score == 1) {				
			//sentence_probabilities.clear();
			sentence_probabilities = vector<precision_type> (current_minibatch_size,0.);
			prop.computeProbsLog(decoder_testing_output_sent_data,
					minibatch_log_likelihood,
					sentence_probabilities);									
			cerr<<endl;
			for (int sent_id=0; sent_id<current_minibatch_size; sent_id++){
				cerr<<"Sequence "<<minibatch_start_index+sent_id<<" probability: "<<sentence_probabilities[sent_id]<<endl;
			}			
			log_likelihood += minibatch_log_likelihood;				
		}
		//writing the predicted sequence
		if (arg_stochastic || arg_greedy) {
			for (int sent_id = 0; sent_id<predicted_sequence.size(); sent_id++) {
				for (int seq_id=0; seq_id<predicted_sequence[sent_id].size()-1 /*we dont want to print </s>*/; seq_id++){
					file << decoder_output_vocab.get_word(predicted_sequence[sent_id][seq_id])<<" ";	
				}
				file<<endl;
			}
		}
		//Now to print the input and output hidden states
		if (arg_hidden_states_file != ""){
			//Then print the hidden 
			for (int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++){
				if (arg_run_lm == 0){
					//hidden_states_file<<"--ENCODER_STATES--"<<endl;
					//Then first generate the hidden from the encoder
					for (int word_index=0; word_index<max_input_sent_len; word_index++){
						if (testing_input_sequence_cont_sent_data(word_index,minibatch_index) == 1){	
							hidden_states_file << "input_symbol: "<<encoder_vocab.get_word(testing_input_sent_data(word_index,minibatch_index))<<endl;
							//now printing all the states
							writeStates(hidden_states_file,
									encoder_h_t,
									encoder_c_t,
									encoder_f_t,
									encoder_i_t,
									encoder_o_t,
									minibatch_index,
									word_index);																											

						}						
					}

					//First get the sentence length
					int current_sentence_length=0;
					for (int word_index=0; word_index<max_output_sent_len; word_index++) {
						if (testing_output_sequence_cont_sent_data(word_index,minibatch_index) == 1)
							current_sentence_length++;	
					}
					current_sentence_length--; //this is because the last word in the output sentence is </s> 
					//The first symbol will be '<s>

					for (int word_index=0; word_index<=current_sentence_length; word_index++){
						if (testing_output_sequence_cont_sent_data(word_index,minibatch_index) == 1){																		
							hidden_states_file << "input_symbol: " << decoder_input_vocab.get_word(decoder_testing_input_sent_data(word_index,minibatch_index))<<endl;
							writeStates(hidden_states_file,
									decoder_h_t,
									decoder_c_t,
									decoder_f_t,
									decoder_i_t,
									decoder_o_t,
									minibatch_index,
									word_index);		
						}

					}
					hidden_states_file<<"<<<<<<<<<NEW SENTENCE>>>>>>>>>"<<endl;
				}
			}
		}

	}//End minibatch  
	 
	 if (arg_predicted_sequence_file != ""){
		 file.close();
	 }
	 if(arg_hidden_states_file != "")
	 	hidden_states_file.close();
	 cerr << "done." << endl;
	 if(arg_score) {
	    cerr << "		Testing log-likelihood base e    :   " << log_likelihood << endl;
		cerr << "		Testing log-likelihood base 2    :   " << log_likelihood/log(2.) << endl;
		cerr<<  "		Testing cross entropy in base 2  :   "<< log_likelihood/(log(2.)*total_output_tokens)<< endl;
		cerr << "         		perplexity               :   "<< exp(-log_likelihood/total_output_tokens) << endl;	 	
	 } 	 


	#ifdef USE_CHRONO
	cerr << "Propagation times:";
	for (int i=0; i<timer.size(); i++)
	  cerr << " " << timer.get(i);
	cerr << endl;
	#endif
	
    return 0;
}
