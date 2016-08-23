//#define EIGEN_NO_MALLOC
//#define EIGEN_NO_AUTOMATIC_RESIZING
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
#include "Rand.h"


//#define EIGEN_DONT_PARALLELIZE


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
//#define EIGEN_NO_AUTOMATIC_RESIZING


int main(int argc, char** argv)
{ 
	cerr<<"precision type is "<<sizeof(precision_type)<<endl;
	srand (time(NULL));
	cerr<<setprecision(16);
    ios::sync_with_stdio(false);
    bool arg_run_lm = 0, arg_carry_states=0, arg_run_tagger=0;
    param myParam;
	int arg_seed;
    try {
      // program options //
      CmdLine cmd("Trains a LSTM neural machine translation model or a language model.", ' ' , "0.1\n","");

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);

      ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 100.", false, 100, "int", cmd);

      ValueArg<precision_type> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "precision_type", cmd);

      ValueArg<precision_type> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 1.", false, 1., "precision_type", cmd);

	  ValueArg<precision_type> fixed_partition_function("", "fixed_partition_function", "Fixed log normalization constant value. Default: 0.", false, 0., "precision_type", cmd);

      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 128.", false, 128, "int", cmd);

      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 128.", false, 128, "int", cmd);

      ValueArg<int> num_epochs("", "num_epochs", "Number of epochs. Default: 10.", false, 10, "int", cmd);

      ValueArg<precision_type> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.1", false, 0.1, "precision_type", cmd);

      ValueArg<precision_type> init_forget("", "init_forget", "value to initialize the bias of the forget gate. Default: 0", false, 0, "precision_type", cmd);

      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0. \
		  (initialize from a uniform distribution)", false, 0, "bool", cmd);

      ValueArg<int> seed("", "seed", "The seed for the random number generator (used for initializing the model parameters). \
		   Default: 1234.", false, 1234, "int", cmd);

      ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: log.", false, "log", "string", cmd);

      ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 64. All gates, cells, hidden layers, \n \
		  							input and output embedding dimension are set to this value", false, 64, "int", cmd);

      ValueArg<int> input_vocab_size("", "input_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);

      ValueArg<int> output_vocab_size("", "output_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);

      ValueArg<string> model_prefix("", "model_prefix", "Prefix for output model files." , true, "", "string", cmd);

	  //TODO: support training with reloaded models
	  //ValueArg<string> load_encoder_file("", "init_encoder_file", "Loading a pre-trained encoder" , false, "", "string", cmd);
	  
	  //ValueArg<string> load_encoder_reverse_file("", "init_encoder_reverse_file", "Loading a pre-trained reverse encoder" , false, "", "string", cmd);

	  //ValueArg<string> load_decoder_file("", "init_decoder_file", "Loading a pre-trained decoder" , false, "", "string", cmd);

      ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);

      ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);

	  ValueArg<string> training_sent_file("", "training_sent_file", "Training sentences file" , true, "", "string", cmd);

	  ValueArg<string> validation_sent_file("", "validation_sent_file", "Validation sentences file" , true, "", "string", cmd);

	  ValueArg<bool> gradient_check("", "gradient_check", "Do you want to do a gradient check or not. 1 = Yes, 0 = No. Default: 0.", false, 0, "bool", cmd);
	  
	  ValueArg<bool> norm_clipping("", "norm_clipping", "Do you want to do norm clipping or gradient clipping. 1 = norm cilpping, \n \
		  			0 = gradient clipping. Default: 1.", false, 1, "bool", cmd);

	  ValueArg<bool> run_lm("", "run_lm", "Run as a language model, \n \
		  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);	

	  ValueArg<bool> run_tagger("", "run_tagger", "Run as a tagger, \n \
	  	  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);		  

	  ValueArg<bool> carry_states("", "carry_states", "Carry the hidden states from one minibatch to another. This option is for \n \
		  			language models only. Carrying hidden states over can improve perplexity. \n \
						1 = yes. Default: 0 (Do not carry hidden states).", false, 0, "bool", cmd);		  

	  ValueArg<bool> reverse_input("", "reverse", "Reverse the input sentence before training, \n \
		  			1 = yes. Default: 0 (No reversing).", false, 0, "bool", cmd);		    

	  ValueArg<bool> use_feed_input("", "use_feed_input", "Linking the attention output to the next hidden state as feed input, \n \
		  			1 = yes. Default: 0 (No feed_input).", false, 0, "bool", cmd);		    

	  ValueArg<precision_type> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "precision_type", cmd);

	  ValueArg<precision_type> dropout_probability("", "dropout_probability", "Dropout probability. Default 0: No dropout", false,0., "precision_type", cmd);
	  
      ValueArg<int> max_epoch("", "max_epoch", "After max_epoch, the learning rate is halved for every subsequent epoch. \n \
		  If not supplied, then the learning rate is modified based on the valdation set. Default: -1", false, -1, "int", cmd);	  

      cmd.parse(argc, argv);

      // define program parameters //
      myParam.input_words_file = input_words_file.getValue();
      myParam.output_words_file = output_words_file.getValue();
	  myParam.training_sent_file = training_sent_file.getValue();
	  myParam.validation_sent_file = validation_sent_file.getValue();
	  myParam.reverse_input = reverse_input.getValue();
	  myParam.use_feed_input = use_feed_input.getValue();
      myParam.model_prefix = model_prefix.getValue();
      myParam.input_vocab_size = input_vocab_size.getValue();
      myParam.output_vocab_size = output_vocab_size.getValue();
      myParam.num_hidden = num_hidden.getValue();
      myParam.loss_function = loss_function.getValue();
      myParam.num_threads = num_threads.getValue();
      myParam.num_noise_samples = num_noise_samples.getValue();
      myParam.minibatch_size = minibatch_size.getValue();
      myParam.validation_minibatch_size = validation_minibatch_size.getValue();
      myParam.num_epochs= num_epochs.getValue();
      myParam.learning_rate = learning_rate.getValue();
	  myParam.adagrad_epsilon = 0;
	  myParam.fixed_partition_function = fixed_partition_function.getValue();
	  myParam.use_momentum = 0;
      myParam.L2_reg = L2_reg.getValue();
      myParam.init_normal= init_normal.getValue();
      myParam.init_range = init_range.getValue();
	  myParam.init_forget = init_forget.getValue();
	  myParam.max_epoch = max_epoch.getValue();
	  myParam.parameter_update = "SGD";
	  myParam.gradient_check = gradient_check.getValue();
	  myParam.norm_clipping = norm_clipping.getValue();
	  myParam.norm_threshold = norm_threshold.getValue();
	  myParam.dropout_probability = dropout_probability.getValue();
	  arg_run_lm = run_lm.getValue();
	  arg_run_tagger = run_tagger.getValue();
	  arg_seed = seed.getValue();
	  arg_carry_states = carry_states.getValue();
	  if (arg_run_lm == 0 && arg_carry_states == 1){
		  cerr<<"--carry_states 1 can only be used with --run_lm 1"<<endl;
		  exit(1);
	  }
	  //myParam.load_encoder_file = load_encoder_file.getVale();
	  //myParam.load_encoder_reverse_file = load_encoder_reverse_file.getVale();
	  //myParam.load_decoder_file = load_decoder_file.getVale();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
	  cerr << training_sent_file.getDescription() << sep << training_sent_file.getValue() << endl;
	  cerr << validation_sent_file.getDescription() << sep << validation_sent_file.getValue() << endl;
      cerr << input_words_file.getDescription() << sep << input_words_file.getValue() << endl;
      cerr << output_words_file.getDescription() << sep << output_words_file.getValue() << endl;
      cerr << model_prefix.getDescription() << sep << model_prefix.getValue() << endl;
      cerr << input_vocab_size.getDescription() << sep << input_vocab_size.getValue() << endl;
      cerr << output_vocab_size.getDescription() << sep << output_vocab_size.getValue() << endl;
	  cerr << norm_clipping.getDescription() << sep << norm_clipping.getValue() <<endl;
	  cerr << norm_threshold.getDescription() << sep << norm_threshold.getValue() <<endl;
	  cerr << dropout_probability.getDescription() << sep << dropout_probability.getValue() <<endl;
	  cerr << gradient_check.getDescription() <<sep <<gradient_check.getValue() <<endl;
	  cerr << max_epoch.getDescription() << sep << max_epoch.getValue() <<endl;
	  cerr << run_lm.getDescription() <<sep <<run_lm.getValue() <<endl;
	  cerr << carry_states.getDescription() <<sep <<carry_states.getValue() <<endl;
	  cerr << reverse_input.getDescription() <<sep <<reverse_input.getValue() <<endl;
	  cerr << use_feed_input.getDescription() <<sep <<use_feed_input.getValue() <<endl;
	  cerr << seed.getDescription() << sep << seed.getValue() <<endl;
	  cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;
	  cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() << endl;
	  cerr << fixed_partition_function.getDescription() << sep << fixed_partition_function.getValue() << endl;
	  //cerr << load_encoder_file.getDescription() <<sep <<load_encoder_file.getValue() <<endl;
	  //cerr << load_encoder_reverse_file.getDescription() <<sep <<load_encoder_reverse_file.getValue() <<endl;
	  //cerr << load_decoder_file.getDescription() <<sep <<load_decoder_file.getValue() <<endl;
	  if (arg_run_lm == 1) {
		  cerr<<"Running as a LSTM language model"<<endl;
	  } else {
		  cerr<<"Running as a LSTM sequence to sequence model"<<endl;
	  }
      cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;
      if (string_to_loss_function(loss_function.getValue()) == InvalidLoss)
      {
	      cerr << "error: invalid loss function: " << loss_function.getValue() << endl;
	      exit(1);
      }
      cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;
      cerr << init_normal.getDescription() << sep << init_normal.getValue() << endl;
      cerr << init_range.getDescription() << sep << init_range.getValue() << endl;
      cerr << num_epochs.getDescription() << sep << num_epochs.getValue() << endl;
      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
      if (myParam.validation_file != "") {
	     cerr << validation_minibatch_size.getDescription() << sep << validation_minibatch_size.getValue() << endl;
      }
      cerr << learning_rate.getDescription() << sep << learning_rate.getValue() << endl;
      cerr << L2_reg.getDescription() << sep << L2_reg.getValue() << endl;
      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);
    int save_threads;

    mt19937 rng(arg_seed), rng_grad_check(arg_seed);

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read training data

	vector< vector<int> > training_input_sent, validation_input_sent, training_sequence_cont_sent;
	vector< vector<int> > training_output_sent, validation_output_sent, validation_sequence_cont_sent;
	vector< vector<int> > decoder_training_input_sent, decoder_training_output_sent;
	vector <vector<int> > decoder_validation_input_sent, decoder_validation_output_sent;
	vector< vector<string> > word_training_input_sent, word_validation_input_sent;
	vector< vector<string> > word_training_output_sent, word_validation_output_sent;

    // Construct vocabulary
    vocabulary input_vocab, output_vocab;
	vocabulary decoder_input_vocab, decoder_output_vocab;

    data_size_t training_data_size, validation_data_size; //num_tokens;
	
	//Reading the input and output sent files
	data_size_t total_output_tokens,
				total_input_tokens,
				total_validation_input_tokens, 
				total_validation_output_tokens, 
				total_training_sequence_tokens,
				total_validation_sequence_tokens;
				
	total_output_tokens = 
		total_input_tokens = 
			total_validation_input_tokens = 
				total_validation_output_tokens = 
					total_training_sequence_tokens = 
						total_validation_sequence_tokens = 0;


	//Even sentences are input
	if( arg_run_lm == 0 ) {
		readEvenSentFile(myParam.training_sent_file, word_training_input_sent, total_input_tokens, 1, 0, myParam.reverse_input); //1: add sentence start and end symbol; 0: indicates input source sentences
		if( myParam.input_words_file == "" ) {
			if( myParam.input_vocab_size )
				createVocabulary(word_training_input_sent, input_vocab, myParam.input_vocab_size); //build encoder vocab according to predefined vocab_size
			else
				createVocabulary(word_training_input_sent, input_vocab); //build encoder vocab using all words in the corpus
			myParam.input_vocab_size = input_vocab.size();
		}
		cerr<<"Input vocab size is "<<myParam.input_vocab_size<<endl;
		integerize(word_training_input_sent, 
						training_input_sent, 
						input_vocab); //transform the word string to word id
    	cerr<<"Number of input tokens "<<total_input_tokens<<endl;						
	}
	
	//Reading output 
	readOddSentFile(myParam.training_sent_file, word_training_output_sent, total_output_tokens,1,1, 0); //We do not reverse the output
				
	int decoder_input_vocab_size;
	int decoder_output_vocab_size;
	decoder_input_vocab_size = decoder_output_vocab_size = 0;
	//If load input model and load output model have been specified
	//After reading the sentence file, create the input and output vocabulary if it hasn't already been specified
	if( myParam.output_words_file == "" ){
		if( myParam.output_vocab_size ) { //build decoder vocab according to predefined vocab_size
			createVocabulary(word_training_output_sent, output_vocab, myParam.output_vocab_size);	
			buildDecoderVocab(word_training_output_sent, 
					decoder_input_vocab,
					0,
					1,
					myParam.output_vocab_size); //0: start index of a sentence; 1: offset to the end of a sentence; decoder input vocab should be from 0 to n-1
			buildDecoderVocab(word_training_output_sent, 
					decoder_output_vocab,
					1,
					0,
					myParam.output_vocab_size); //decoder output decoder should be from 1 to n of a sentence
		}
		else {
			createVocabulary(word_training_output_sent, output_vocab);	
			buildDecoderVocab(word_training_output_sent, 
					decoder_input_vocab,
					0,
					1);
			buildDecoderVocab(word_training_output_sent, 
					decoder_output_vocab,
					1,
					0);					

		}
		myParam.output_vocab_size = output_vocab.size();
		decoder_input_vocab_size = decoder_input_vocab.size();
		decoder_output_vocab_size = decoder_output_vocab.size();
	}
	cerr<<"Decoder input vocab size is "<<decoder_input_vocab_size<<endl;
	cerr<<"Decoder output vocab size is "<<decoder_output_vocab_size<<endl;
				
	//Creating separate decoder input vocab and decoder output vocab

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

	data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;

	cerr<<"Number of output tokens "<<total_output_tokens<<endl;
	cerr<<"Number of minibatches "<<num_batches<<endl;
    cerr << "Number of training instances "<< decoder_training_output_sent.size() << endl;
    cerr << "Number of validation instances "<< decoder_validation_output_sent.size() << endl;
	
    Matrix<int,Dynamic,Dynamic> training_data;
	
    // Read validation data
    validation_data_size = 0;
	if( myParam.validation_sent_file != "" ) {
		if (arg_run_lm == 0) {
			readEvenSentFile(myParam.validation_sent_file, 
						word_validation_input_sent, 
						total_validation_input_tokens,
						1,
						0,
						myParam.reverse_input);
			//integerizing the validation data
			integerize(word_validation_input_sent, 
							validation_input_sent, 
							input_vocab);			
			cerr<<"Validation input tokens "<<total_validation_input_tokens<<endl;									
		}
		readOddSentFile(myParam.validation_sent_file, 
					word_validation_output_sent, 
					total_validation_output_tokens,
					1,
					1,
					0);

		integerize(word_validation_output_sent, 
						validation_output_sent, 
						output_vocab);	
						
		integerize(word_validation_output_sent, 
				   decoder_validation_output_sent,
				   decoder_output_vocab,
				   1,
				   0);
	   	integerize(word_validation_output_sent, 
	   			   decoder_validation_input_sent,
	   			   decoder_input_vocab,
	   			   0,
	   			   1);			
				      						
		cerr<<"Validation output tokens "<<total_validation_output_tokens<<endl;									
	}
	
	validation_data_size = validation_output_sent.size();

    ///// Read in vocabulary file. We don't actually use it; it just gets reproduced in the output file
    vector<string> input_words;
    if( myParam.input_words_file != "" )
    {
        readWordsFile(myParam.input_words_file, input_words);
		if( myParam.input_vocab_size == 0 )
			myParam.input_vocab_size = input_words.size();
    }

    vector<string> output_words;
    if( myParam.output_words_file != "" )
    {
        readWordsFile(myParam.output_words_file, output_words);
		if (myParam.output_vocab_size == 0)
			myParam.output_vocab_size = output_words.size();
    }

    ///// Create and initialize the neural network and associated propagators.
	myParam.input_embedding_dimension = myParam.num_hidden;
	myParam.output_embedding_dimension = myParam.num_hidden;
	
    model nn, nn_reverse; //left-to-right LSTM encoder model and right-to-left LSTM encoder model, which will be shared in each position
    nn.resize(myParam.ngram_size,
        myParam.input_vocab_size,
        myParam.input_vocab_size,
        myParam.input_embedding_dimension,
        myParam.num_hidden,
        myParam.output_embedding_dimension);

    nn.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
		-log(myParam.output_vocab_size),
        myParam.init_forget,
        myParam.parameter_update,
        myParam.adagrad_epsilon);	

	nn_reverse.resize(myParam.ngram_size,
        myParam.input_vocab_size,
        myParam.input_vocab_size,
        myParam.input_embedding_dimension,
        myParam.num_hidden,
        myParam.output_embedding_dimension);

    nn_reverse.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
		-log(myParam.output_vocab_size),
        myParam.init_forget,
        myParam.parameter_update,
        myParam.adagrad_epsilon);	

	//Creating the input node, the input node maps the word id to word embedding and then forward it to the input of a LSTM
	google_input_model input(myParam.num_hidden, 
						myParam.input_vocab_size,
						myParam.input_embedding_dimension);
	input.resize(myParam.input_vocab_size,
	    myParam.input_embedding_dimension,
	    myParam.num_hidden);

	input.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
        myParam.parameter_update,
        myParam.adagrad_epsilon);
	
	nn.set_input(input);
	nn_reverse.set_input(input); //the left-to-right and right-to-left encoders share the same word embedding
				
	model nn_decoder; //the LSTM decoder
	nn_decoder.set_feed_input_flag(myParam.use_feed_input);
	nn_decoder.resize(myParam.ngram_size,
			decoder_input_vocab_size,
			decoder_output_vocab_size,
			myParam.input_embedding_dimension*2,
			myParam.num_hidden*2,
			myParam.output_embedding_dimension*2); //since we concatenate the left-to-right and right-to-left encoding, the decoder dim is set twice as much

	    nn_decoder.initialize(rng,
	        myParam.init_normal,
	        myParam.init_range,
			-log(myParam.output_vocab_size),
	        myParam.init_forget,
	        myParam.parameter_update,
	        myParam.adagrad_epsilon);		
		//Creating the input node
		google_input_model decoder_input(myParam.num_hidden*2, 
							decoder_input_vocab_size,
							myParam.input_embedding_dimension*2);
		decoder_input.resize(decoder_input_vocab_size,
		    myParam.input_embedding_dimension*2,
		    myParam.num_hidden*2);

		decoder_input.initialize(rng,
	        myParam.init_normal,
	        myParam.init_range,
	        myParam.parameter_update,
	        myParam.adagrad_epsilon);

		nn_decoder.set_input(decoder_input);

		//creating feed-input node, which links the attention output as another input to the next LSTM node
		hidden_to_hidden_input_model feed_input(myParam.num_hidden*2,
				decoder_input_vocab_size,
				myParam.input_embedding_dimension*2,
				myParam.norm_threshold);
		feed_input.resize(decoder_input_vocab_size,
				myParam.input_embedding_dimension*2,
				myParam.num_hidden*2);

		feed_input.initialize(rng,
				myParam.init_normal,
				myParam.init_range,
				myParam.parameter_update,
				myParam.adagrad_epsilon);
		if( nn_decoder.feed_input_flag ) //if use_feed_input, activate the feed input in decoder LSTM
			nn_decoder.set_feed_input(feed_input);

    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL: TODO
	
    if( myParam.model_file != "" ) {
		nn.read(myParam.model_file);
		cerr<<"reading the model"<<endl;
	} else {
	  rng_grad_check = rng; //The range for gradient check should have exactly the same state as rng for the NCE gradient checking to work
	  
    }

    //loss function can be log-likelihood or noisy contrastive estimation (NCE), for large-scale data, NCE is necessary
    loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

	//using left-to-right, right-to-left LSTM encoders and LSTM decoder to initialize the propagator for training 
    propagator<Google_input_node, Hidden_to_hidden_input_node, google_input_model, hidden_to_hidden_input_model> prop(nn, nn_reverse, 
														nn_decoder, 
														myParam.minibatch_size);
	if (myParam.dropout_probability > 0 ){
		prop.resizeDropout(myParam.minibatch_size, myParam.dropout_probability);
	}

	//using left-to-right, right-to-left LSTM encoders and LSTM decoder to initialize the propagator for validation 
    propagator<Google_input_node, Hidden_to_hidden_input_node, google_input_model, hidden_to_hidden_input_model> prop_validation(nn, nn_reverse,
															nn_decoder, 
															myParam.validation_minibatch_size);
														    vector<data_size_t> unigram_counts;

	//if we use NCE loss, we should prepare the noise sampling distribution. Here, we use unigram distribution to sample noise
	multinomial<data_size_t> unigram;																
	if (loss_function == NCELoss){
	    vector<data_size_t> unigram_counts = vector<data_size_t>(decoder_output_vocab_size,0);
		for (int sent_index=0; sent_index<decoder_training_output_sent.size(); sent_index++){
			for (int word_index=0; word_index<decoder_training_output_sent[sent_index].size(); word_index++){
				unigram_counts[decoder_training_output_sent[sent_index][word_index]] += 1;
			}
		}
		unigram = multinomial<data_size_t> (unigram_counts);
	}
	if (loss_function == NCELoss)
		prop.resizeNCE(myParam.num_noise_samples, myParam.fixed_partition_function, unigram);
																

    ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

	Rand rnd_instance; //for training data shuffle
    cerr<<"Number of training minibatches: "<<num_batches<<endl;
	
    int num_validation_batches = 0;
    if (validation_data_size > 0)
    {
        num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
		cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
    } 
	
    precision_type current_momentum = myParam.initial_momentum;
    precision_type momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
    precision_type current_learning_rate = myParam.learning_rate;
    precision_type current_validation_ll = 0.0;

    int input_vocab_size = myParam.input_vocab_size;
    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
    int validation_minibatch_size = myParam.validation_minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;

	precision_type best_perplexity = 999999999;
	int best_model = 0;	

	//Resetting the gradient in the beginning
	prop.resetGradient();

	//initialize the source context representation
	vector< Matrix<precision_type,Dynamic,Dynamic> > source_ctx(105, Matrix<precision_type,Dynamic,Dynamic>()); //only allow for max 105 input words
        vector< Matrix<precision_type,Dynamic,Dynamic> > source_cell(105, Matrix<precision_type,Dynamic,Dynamic>());
        for( int src_id = 0; src_id < 105; src_id++ ) {
                source_ctx[src_id].setZero(myParam.num_hidden*2, minibatch_size);
                source_cell[src_id].setZero(myParam.num_hidden*2, minibatch_size);
        }

    for( int epoch=0; epoch<myParam.num_epochs; epoch++ )
    { 
		//shuffle the data in each epoch
		rnd_instance.shuffle(training_input_sent, decoder_training_input_sent, decoder_training_output_sent);

        cerr << "Epoch " << epoch+1 << endl;
        cerr << "Current learning rate: " << current_learning_rate << endl;

        if (myParam.use_momentum) 
			cerr << "Current momentum: " << current_momentum << endl;
		else
	            current_momentum = -1;

		cerr << "Training minibatches: ";

		precision_type log_likelihood = 0.0;

		int num_samples = 0;
		if( loss_function == LogLoss )
		    num_samples = output_vocab_size;
		else if( loss_function == NCELoss )
		    num_samples = 1+num_noise_samples;
	
	    data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
		precision_type data_log_likelihood=0;	
		//initialize some variables
		Matrix<precision_type,Dynamic,Dynamic> current_c_for_gradCheck, current_h_for_gradCheck, current_c,current_h, init_c, init_h, init_ctx_mean, init_cell_mean;
		init_c.setZero(myParam.num_hidden,minibatch_size);
		init_h.setZero(myParam.num_hidden,minibatch_size);
		current_c.setZero(myParam.num_hidden, minibatch_size);
		current_h.setZero(myParam.num_hidden, minibatch_size);			
		init_ctx_mean.setZero(myParam.num_hidden*2, minibatch_size);
		init_cell_mean.setZero(myParam.num_hidden*2, minibatch_size);			

		//cerr<<"About to start training "<<endl;
		for( data_size_t batch=0; batch < num_batches; batch++ )
		{
			if (arg_carry_states == 0) {
				current_c.setZero(myParam.num_hidden, minibatch_size);
				current_h.setZero(myParam.num_hidden, minibatch_size);			
			}
			if (batch > 0 && batch % 100 == 0)
			{
				cerr << batch <<"...";
			} 

			//determine minibatch start index and end index
			data_size_t minibatch_start_index = minibatch_size * batch;
			data_size_t minibatch_end_index = min(training_data_size-1, static_cast<data_size_t> (minibatch_start_index+minibatch_size-1));
			int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), training_data_size - minibatch_start_index);

			//Getting a minibatch of sentences
			vector<int> minibatch_input_sentences, minibatch_input_sentences_reverse, minibatch_output_sentences, minibatch_sequence_cont_sentences, minibatch_input_token_num;
			vector<int> minibatch_input_sequence_cont_sentences, minibatch_input_sequence_cont_sentences_reverse, minibatch_output_sequence_cont_sentences;
			vector<int> minibatch_decoder_output_sentences, minibatch_decoder_input_sentences;
			unsigned int max_input_sent_len, max_output_sent_len;
			unsigned int minibatch_output_tokens,minibatch_input_tokens, minibatch_sequence_cont_tokens;
			minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;
			max_input_sent_len = max_output_sent_len = 0;

			if (myParam.gradient_check) {
				current_c_for_gradCheck = current_c;
				current_h_for_gradCheck = current_h;
				cerr<<"current_c_for_gradCheck "<<current_c_for_gradCheck<<endl;
				cerr<<"current_h_for_gradCheck "<<current_h_for_gradCheck<<endl;
			}													
			init_c = current_c;
			init_h = current_h; 	

			if (arg_run_lm == 0) {
				//select the training_input_sent from minibatch_start_index to minibatch_end_index to form minibatch_input_sentences and minibatch_input_sentences_reverse; minibatch_input_token_num stores the token number for each sentence in the batch; max_input_sent_len keeps the sentence max_length in the minibatch; minibatch_input_tokens stores the total token number of the minibatch; 1 stands for data (0 stands for continuation flags)
				miniBatchifyEncoder(training_input_sent, 
						minibatch_input_sentences,
						minibatch_input_sentences_reverse,
						minibatch_input_token_num,
						minibatch_start_index,
						minibatch_end_index,
						max_input_sent_len,
						minibatch_input_tokens,
						1);	
				minibatch_input_tokens = 0;
				miniBatchifyEncoder(training_input_sent, 
						minibatch_input_sequence_cont_sentences,
						minibatch_input_sequence_cont_sentences_reverse,
						minibatch_input_token_num,
						minibatch_start_index,
						minibatch_end_index,
						max_input_sent_len,
						minibatch_input_tokens,
						0);			
				//convert the minibatch data and continuation vectors into matrix
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data (minibatch_input_sentences.data(), 
						max_input_sent_len,
						current_minibatch_size);
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data (minibatch_input_sequence_cont_sentences.data(),
						max_input_sent_len,
						current_minibatch_size);	
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data_reverse (minibatch_input_sentences_reverse.data(), 
						max_input_sent_len,
						current_minibatch_size);
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data_reverse (minibatch_input_sequence_cont_sentences_reverse.data(),
						max_input_sent_len,
						current_minibatch_size);	
				if (myParam.dropout_probability > 0.) {
					//encode the source sentences with left-to-right LSTM
					prop.fPropEncoderDropout(training_input_sent_data,
							current_c,
							current_h,
							training_input_sequence_cont_sent_data,
							rng);					
					current_c = init_c;
					current_h = init_h;
					//encode the source sentences with right-to-left LSTM
					prop.fPropEncoderDropoutReverse(training_input_sent_data_reverse,
							current_c,
							current_h,
							training_input_sequence_cont_sent_data_reverse,
							rng);					
				} else {
					prop.fPropEncoder(training_input_sent_data,
							current_c,
							current_h,
							training_input_sequence_cont_sent_data);
					current_c = init_c;
					current_h = init_h;
					prop.fPropEncoderReverse(training_input_sent_data_reverse,
							current_c,
							current_h,
							training_input_sequence_cont_sent_data_reverse);
				}																																									
			}

			//concatenate the forward and backward source contexts
			#pragma omp parallel for
			for( int i = 0; i < max_input_sent_len; ++i ){
				int j = 0;
				for( j = 0; j < current_minibatch_size; ++j ){
					//minibatch_input_token_num[j] stores the actural token number for the jth sentence in the minibatch
					//set the padded words in the beginning and initialize them zeros
					int padded_word_num = max_input_sent_len - minibatch_input_token_num[j];
					if( i < padded_word_num ) {
						source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
							prop.encoder_lstm_nodes_reverse[i].h_t.col(j);
						source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
							prop.encoder_lstm_nodes_reverse[i].c_t.col(j);
					}
					else {
						int reverse_id = max_input_sent_len-i+padded_word_num-1; //reverse_id in right_to_left encoder points at the same word as i in the left_to_right encoder 
						source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
							prop.encoder_lstm_nodes_reverse[reverse_id].h_t.col(j);
						source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
							prop.encoder_lstm_nodes_reverse[reverse_id].c_t.col(j);
					}
				}
				for( ; j < minibatch_size; ++j ) {
					source_ctx[i].col(j) << prop.encoder_lstm_nodes[i].h_t.col(j),
						prop.encoder_lstm_nodes_reverse[i].h_t.col(j);
					source_cell[i].col(j) << prop.encoder_lstm_nodes[i].c_t.col(j),
						prop.encoder_lstm_nodes_reverse[i].c_t.col(j);
				}
			}

			//use the average context as the initial context of the decoder
			Matrix<precision_type,Dynamic,Dynamic> source_ctx_mean = init_ctx_mean;
			Matrix<precision_type,Dynamic,Dynamic> source_cell_mean = init_cell_mean;
			for( int i = 0; i < max_input_sent_len; ++i ){
				source_ctx_mean += source_ctx[i];
				source_cell_mean += source_cell[i];
			}
			#pragma omp parallel for
			for( int i = 0; i < current_minibatch_size; ++i ){
				source_ctx_mean.col(i) = source_ctx_mean.col(i)/minibatch_input_token_num[i];
				source_cell_mean.col(i) = source_cell_mean.col(i)/minibatch_input_token_num[i];
			}


			//preparing the data info for the decoder
			//extract the decoder_training_output_sent from minibatch_start_index to minibatch_end_index to minibatch_decoder_output_sentences; max_output_sent_len keeps the sentence max length in the minibatch; minibatch_output_tokens stores the total token number in the minibatch; 1 stands for preparing data; -1 indicates the padded value
			miniBatchifyDecoder(decoder_training_output_sent, 
					minibatch_decoder_output_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					1,
					-1);
			minibatch_output_tokens =0;						
			miniBatchifyDecoder(decoder_training_input_sent, 
					minibatch_decoder_input_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					1,
					0);									
			//convert vectors to maxtrix
			Map< Matrix<int, Dynamic, Dynamic> >  decoder_training_output_sent_data (minibatch_decoder_output_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);
			Map< Matrix<int, Dynamic, Dynamic> >  decoder_training_input_sent_data (minibatch_decoder_input_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);	

			//preparing the continuation vectors and convert them to matrix
			minibatch_output_tokens =0;	
			miniBatchifyDecoder(decoder_training_input_sent, 
					minibatch_output_sequence_cont_sentences,
					minibatch_start_index,
					minibatch_end_index,
					max_output_sent_len,
					minibatch_output_tokens,
					0,
					0);	
			Map< Array<int,Dynamic,Dynamic> > training_output_sequence_cont_sent_data(minibatch_output_sequence_cont_sentences.data(),
					max_output_sent_len,
					current_minibatch_size);

			if (myParam.dropout_probability > 0.) {
				//decode the minibatch sentences with LSTM: decoder_training_input_sent_data is the data input of LSTM; source_cell_mean and source_ctx_mean serve as the initial history of the decoder LSTM; source_ctx is used for attention model; minibatch_input_token_num stores the actural token number for each sentence in the minibatch
				prop.fPropDecoderDropout(decoder_training_input_sent_data,
						source_cell_mean,
						source_ctx_mean,
						source_ctx,
						training_output_sequence_cont_sent_data,
						minibatch_input_token_num,
						max_input_sent_len,
						rng);				
			} else {
				prop.fPropDecoder(decoder_training_input_sent_data,
						source_cell_mean,
						source_ctx_mean,
						source_ctx,
						training_output_sequence_cont_sent_data,
						minibatch_input_token_num,
						max_input_sent_len);
			}

			precision_type adjusted_learning_rate = current_learning_rate;
			if (!myParam.norm_clipping){
				adjusted_learning_rate /= current_minibatch_size;			
			}

			//computing losses
			if (myParam.dropout_probability > 0) {
				//compute the losses according to the decoder output
				prop.computeLossesDropout(decoder_training_output_sent_data,
						data_log_likelihood,
						myParam.gradient_check,
						myParam.norm_clipping,
						loss_function,
						unigram,
						num_noise_samples,
						rng);
				//do decoder LSTM backpropagation after obtaining loss
				prop.bPropDecoderDropout(decoder_training_input_sent_data,
						source_ctx,
						minibatch_input_token_num,
						max_input_sent_len,
						myParam.gradient_check,
						myParam.norm_clipping); 
			} else {
				prop.computeLosses(decoder_training_output_sent_data,
						data_log_likelihood,
						myParam.gradient_check,
						myParam.norm_clipping,
						loss_function,
						unigram,
						num_noise_samples,
						rng);
				prop.bPropDecoder(decoder_training_input_sent_data,
						source_ctx,
						minibatch_input_token_num,
						max_input_sent_len,
						myParam.gradient_check,						  
						myParam.norm_clipping);
			}	


			if (arg_run_lm == 0) { 
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data (minibatch_input_sentences.data(), 
						max_input_sent_len,
						current_minibatch_size);
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data (minibatch_input_sequence_cont_sentences.data(),
						max_input_sent_len,
						current_minibatch_size);						
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data_reverse (minibatch_input_sentences_reverse.data(), 
						max_input_sent_len,
						current_minibatch_size);
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data_reverse (minibatch_input_sequence_cont_sentences_reverse.data(),
						max_input_sent_len,
						current_minibatch_size);						
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

			if (myParam.gradient_check) {		
				//for gradient check
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data (minibatch_input_sentences.data(), 
						max_input_sent_len,
						current_minibatch_size);		
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data (minibatch_input_sequence_cont_sentences.data(),
						max_input_sent_len,
						current_minibatch_size);
				Map< Matrix<int,Dynamic,Dynamic> > training_input_sent_data_reverse (minibatch_input_sentences_reverse.data(), 
						max_input_sent_len,
						current_minibatch_size);		
				Map< Array<int,Dynamic,Dynamic> > training_input_sequence_cont_sent_data_reverse (minibatch_input_sequence_cont_sentences_reverse.data(),
						max_input_sent_len,
						current_minibatch_size);																		 
				prop.gradientCheck(training_input_sent_data,
						training_input_sent_data_reverse,
						decoder_training_input_sent_data,
						decoder_training_output_sent_data,
						current_c_for_gradCheck,
						current_h_for_gradCheck,
						unigram,
						num_noise_samples,
						rng_grad_check,
						loss_function,
						training_input_sequence_cont_sent_data,
						training_input_sequence_cont_sent_data_reverse,
						training_output_sequence_cont_sent_data,
						minibatch_input_token_num,
						arg_run_lm,
						myParam.dropout_probability);
				//for the next minibatch, we want the range to be updated as well
				rng_grad_check = rng;
			}

			//Updaging using local grad norms and SGD
			prop.updateParams(adjusted_learning_rate,
					max_input_sent_len + max_output_sent_len,
					current_momentum,
					myParam.L2_reg,
					myParam.norm_clipping,
					myParam.norm_threshold,
					loss_function,
					arg_run_lm);	

			//Resetting the gradients
			prop.resetGradient();

			if( batch == num_batches/2 || batch == num_batches-1 ) { //test the validation set every half epoch
				#ifdef USE_CHRONO
				cerr << "Propagation times:";
				for (int i=0; i<timer.size(); i++)
					cerr << " " << timer.get(i);
				cerr << endl;
				#endif

				//scale the model before writing it
				input.scale(1.-myParam.dropout_probability);
				decoder_input.scale(1.-myParam.dropout_probability);
				nn_decoder.scale(1.-myParam.dropout_probability);

				if (epoch % 1 == 0 && validation_data_size > 0)
				{
					cerr<<"Computing validation perplexity..."<<endl;
					//////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
					////////////////////////////////////////////////////////////////

					precision_type validation_log_likelihood = 0.0;

					Matrix<precision_type,Dynamic,Dynamic> current_validation_c, current_validation_h, validation_init_c, validation_init_h;
					validation_init_c.setZero(myParam.num_hidden, validation_minibatch_size);
					validation_init_h.setZero(myParam.num_hidden, validation_minibatch_size);
					current_validation_c.setZero(myParam.num_hidden, validation_minibatch_size);
					current_validation_h.setZero(myParam.num_hidden, validation_minibatch_size);

					for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
					{
						if (arg_carry_states == 0) {
							current_validation_c.setZero(myParam.num_hidden, validation_minibatch_size);
							current_validation_h.setZero(myParam.num_hidden, validation_minibatch_size);
						}
						precision_type validation_minibatch_log_likelihood = 0.;
						data_size_t validation_minibatch_start_index = validation_minibatch_size * validation_batch;
						data_size_t validation_minibatch_end_index = min(validation_data_size-1, static_cast<data_size_t> (validation_minibatch_start_index+validation_minibatch_size-1));

						int current_validation_minibatch_size = min(static_cast<data_size_t>(validation_minibatch_size), validation_data_size - validation_minibatch_start_index);

						//Taking the input and output sentence and setting the validation data to it.
						//Getting a minibatch of sentences
						vector<int> validation_minibatch_input_sentences, 
							validation_minibatch_input_sentences_reverse,
							validation_minibatch_output_sentences, 
							validation_minibatch_input_sequence_cont_sentences,
							validation_minibatch_input_sequence_cont_sentences_reverse,
							validation_minibatch_output_sequence_cont_sentences,
							validation_minibatch_decoder_output_sentences,
							validation_minibatch_decoder_input_sentences;

						unsigned int validation_max_input_sent_len, validation_max_output_sent_len;
						unsigned int validation_minibatch_output_tokens, validation_minibatch_input_tokens, validation_minibatch_sequence_cont_tokens;
						vector<int> validation_minibatch_input_token_num;
						validation_minibatch_output_tokens = validation_minibatch_input_tokens = validation_minibatch_sequence_cont_tokens = 0;	
						validation_max_input_sent_len = validation_max_output_sent_len = 0;

						if (arg_run_lm == 0) {
							miniBatchifyEncoder(validation_input_sent, 
									validation_minibatch_input_sentences,
									validation_minibatch_input_sentences_reverse,
									validation_minibatch_input_token_num,
									validation_minibatch_start_index,
									validation_minibatch_end_index,
									validation_max_input_sent_len,
									validation_minibatch_input_tokens,
									1);	
							validation_minibatch_input_tokens = 0;
							miniBatchifyEncoder(validation_input_sent, 
									validation_minibatch_input_sequence_cont_sentences,
									validation_minibatch_input_sequence_cont_sentences_reverse,
									validation_minibatch_input_token_num,
									validation_minibatch_start_index,
									validation_minibatch_end_index,
									validation_max_input_sent_len,
									validation_minibatch_input_tokens,
									0);		
							Map< Matrix<int,Dynamic,Dynamic> > validation_input_sent_data (validation_minibatch_input_sentences.data(), 
									validation_max_input_sent_len,
									current_validation_minibatch_size);
							Map< Array<int,Dynamic,Dynamic> > validation_input_sequence_cont_sent_data (validation_minibatch_input_sequence_cont_sentences.data(),
									validation_max_input_sent_len,
									current_validation_minibatch_size);	
							Map< Matrix<int,Dynamic,Dynamic> > validation_input_sent_data_reverse (validation_minibatch_input_sentences_reverse.data(), 
									validation_max_input_sent_len,
									current_validation_minibatch_size);
							Map< Array<int,Dynamic,Dynamic> > validation_input_sequence_cont_sent_data_reverse (validation_minibatch_input_sequence_cont_sentences_reverse.data(),
									validation_max_input_sent_len,
									current_validation_minibatch_size);	
							//encoding the source sentence from left to right and from right to left
							prop_validation.fPropEncoder(validation_input_sent_data,
									current_validation_c,
									current_validation_h,
									validation_input_sequence_cont_sent_data);																															
							current_validation_c = validation_init_c;
							current_validation_h = validation_init_h;
							prop_validation.fPropEncoderReverse(validation_input_sent_data_reverse,
									current_validation_c,
									current_validation_h,
									validation_input_sequence_cont_sent_data_reverse);
						}				

						//concatenate the forward and backward source contexts
						#pragma omp parallel for
						for( int i = 0; i < validation_max_input_sent_len; ++i ){
							for( int j = 0; j < current_validation_minibatch_size; ++j ){
								int padded_word_num = validation_max_input_sent_len - validation_minibatch_input_token_num[j];
								if( i < padded_word_num ) {
									source_ctx[i].col(j) << prop_validation.encoder_lstm_nodes[i].h_t.col(j),
										prop_validation.encoder_lstm_nodes_reverse[i].h_t.col(j);
									source_cell[i].col(j) << prop_validation.encoder_lstm_nodes[i].c_t.col(j),
										prop_validation.encoder_lstm_nodes_reverse[i].c_t.col(j);
								}
								else {
									source_ctx[i].col(j) << prop_validation.encoder_lstm_nodes[i].h_t.col(j),
										prop_validation.encoder_lstm_nodes_reverse[validation_max_input_sent_len-i+padded_word_num-1].h_t.col(j);
									source_cell[i].col(j) << prop_validation.encoder_lstm_nodes[i].c_t.col(j),
										prop_validation.encoder_lstm_nodes_reverse[validation_max_input_sent_len-i+padded_word_num-1].c_t.col(j);
								}
							}
						}

						source_ctx_mean = init_ctx_mean;
						source_cell_mean = init_cell_mean;
						for( int i = 0; i < validation_max_input_sent_len; ++i ){
							source_ctx_mean += source_ctx[i];
							source_cell_mean += source_cell[i];
						}

						#pragma omp parallel for
						for( int i = 0; i < current_validation_minibatch_size; ++i ){
							source_ctx_mean.col(i) = source_ctx_mean.col(i)/validation_minibatch_input_token_num[i];
							source_cell_mean.col(i) = source_cell_mean.col(i)/validation_minibatch_input_token_num[i];
						}


						miniBatchifyDecoder(decoder_validation_output_sent, 
								validation_minibatch_decoder_output_sentences,
								validation_minibatch_start_index,
								validation_minibatch_end_index,
								validation_max_output_sent_len,
								validation_minibatch_output_tokens,
								1,
								-1);		

						miniBatchifyDecoder(decoder_validation_input_sent, 
								validation_minibatch_decoder_input_sentences,
								validation_minibatch_start_index,
								validation_minibatch_end_index,
								validation_max_output_sent_len,
								validation_minibatch_output_tokens,
								1,
								0);			
						Map< Matrix<int, Dynamic, Dynamic> >  decoder_validation_output_sent_data (validation_minibatch_decoder_output_sentences.data(),
								validation_max_output_sent_len,
								current_validation_minibatch_size);
						Map< Matrix<int, Dynamic, Dynamic> > decoder_validation_input_sent_data (validation_minibatch_decoder_input_sentences.data(),
								validation_max_output_sent_len,
								current_validation_minibatch_size);		

						miniBatchifyDecoder(decoder_validation_input_sent, 
								validation_minibatch_output_sequence_cont_sentences,
								validation_minibatch_start_index,
								validation_minibatch_end_index,
								validation_max_output_sent_len,
								validation_minibatch_output_tokens,
								0);		

						Map< Array<int,Dynamic,Dynamic> > validation_output_sequence_cont_sent_data (validation_minibatch_output_sequence_cont_sentences.data(),
								validation_max_output_sent_len,
								current_validation_minibatch_size);

						prop_validation.fPropDecoder(decoder_validation_input_sent_data,
								source_cell_mean,
								source_ctx_mean,
								source_ctx,
								validation_output_sequence_cont_sent_data,
								validation_minibatch_input_token_num,
								validation_max_input_sent_len);	

						prop_validation.computeProbsLog(decoder_validation_output_sent_data,
								validation_minibatch_log_likelihood);
						validation_log_likelihood += validation_minibatch_log_likelihood;
					}

					cerr << "		Per symbol validation probability           "<<epoch<<":      " << exp(validation_log_likelihood)/total_validation_output_tokens << endl;
					cerr << "		Validation log-likelihood base e in epoch   "<<epoch<<":      " << validation_log_likelihood << endl;
					cerr << "		Validation log-likelihood base 2 in epoch   "<<epoch<<":      " << validation_log_likelihood/log(2.) << endl;
					cerr<<  "		Validation cross entropy in base 2 in epoch "<<epoch<<":      "<< validation_log_likelihood/(log(2.)*total_validation_output_tokens)<< endl;
					cerr << "         		perplexity in epoch                 "<<epoch<<":      "<< exp(-validation_log_likelihood/total_validation_output_tokens) << endl;

					// If the validation perplexity decreases, halve the learning rate.
					if( exp(-validation_log_likelihood/total_validation_output_tokens) < best_perplexity ){

						cerr<<"Perplexity on validation improved." <<endl;
						cerr<<"Previous best perplexity from epoch "<<best_model<<" was "<<best_perplexity<<endl;
						best_perplexity = exp(-validation_log_likelihood/total_validation_output_tokens);
						//only write the best model
						if (myParam.model_prefix != "")
						{
							cerr << "Overwriting the previous best model from epoch " << best_model<< endl;

							nn_decoder.write(myParam.model_prefix + ".decoder.best", decoder_input_vocab.words(), decoder_output_vocab.words());
							if (arg_run_lm == 0) {
								nn.write(myParam.model_prefix + ".encoder.best" , input_vocab.words(), output_vocab.words());					
								nn_reverse.write(myParam.model_prefix + ".encoder_reverse.best" , input_vocab.words(), output_vocab.words());					
							}

							best_model = epoch+1;
						}				
					}
					//scale the model back for training
					//scale the model before writing it
					input.scale(1./(1.-myParam.dropout_probability));
					decoder_input.scale(1./(1-myParam.dropout_probability));
					nn_decoder.scale(1./(1-myParam.dropout_probability));
					if( myParam.max_epoch > -1 && epoch+1 >= myParam.max_epoch ) {
						current_learning_rate /= 2;
					} else  if( epoch > 0 && validation_log_likelihood < current_validation_ll && myParam.parameter_update != "ADA" )
					{ 
						if( current_learning_rate > 0.2 )
							current_learning_rate /= 2;
					}
					current_validation_ll = validation_log_likelihood;
				}

				if( batch == num_batches-1 ) {
					cerr << "done." << endl;
					if (loss_function == LogLoss)
					{
						//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
						//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
						//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
						//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
						//log_likelihood /= sent_len;		
						cerr << "Per symbol training probability      " << exp(data_log_likelihood)/total_output_tokens << endl;
						cerr << "Training log-likelihood base e:      " << data_log_likelihood << endl;
						cerr << "Training log-likelihood base 2:      " << data_log_likelihood/log(2.) << endl;
						cerr << "Training cross entropy in base 2 is  " <<data_log_likelihood/(log(2.)*total_output_tokens)<< endl;
						cerr << "         perplexity:                 " << exp(-data_log_likelihood/total_output_tokens) << endl;
					}
					else if (loss_function == NCELoss) 
					{
						cerr << "Training NCE log-likelihood: " << data_log_likelihood << endl;
						cerr << "Average NCE log-likelihood " << data_log_likelihood/(total_output_tokens*myParam.num_noise_samples) << endl;
					}

					if (myParam.use_momentum)
						current_momentum += momentum_delta;
				}
			}

		}

	}
	cerr<<" The best validation perplexity achieved in epoch "<<best_model<<" was "<<best_perplexity<<" and the models are ";
		if (arg_run_lm == 1) {
			cerr<<myParam.model_prefix<<".decoder.best"<<endl;
		} else {
			cerr<<myParam.model_prefix<<".encoder.best"<<", "<<myParam.model_prefix<<".encoder_reverse.best"<<", "<<myParam.model_prefix<<".decoder.best"<<endl;
		}
    return 0;
}
