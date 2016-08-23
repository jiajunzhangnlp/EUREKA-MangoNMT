#pragma once

#include <string>

namespace nplm
{

struct param 
{
    std::string train_file;
    std::string validation_file;
    std::string test_file;

    std::string model_file;
	std::string encoder_model_file;
	std::string encoder_reverse_model_file;
	std::string decoder_model_file;

    std::string unigram_probs_file;
    std::string words_file;
    std::string input_words_file;
    std::string output_words_file;
    std::string model_prefix;
	std::string input_sent_file;
	std::string output_sent_file;
	std::string input_validation_sent_file;
	std::string output_validation_sent_file;
	std::string training_sequence_cont_file;
	std::string testing_sequence_cont_file;
	std::string validation_sequence_cont_file;

    int ngram_size;
    int vocab_size;
    int input_vocab_size;
    int output_vocab_size;
    int num_hidden;
    int embedding_dimension;
    int input_embedding_dimension;
    int output_embedding_dimension;
    std::string activation_function;
    std::string loss_function;
    std::string parameter_update;

    int minibatch_size;
    int validation_minibatch_size;
    int num_epochs;
    precision_type learning_rate;
    precision_type conditioning_constant;
    precision_type decay;
    precision_type adagrad_epsilon;
	precision_type fixed_partition_function;
    bool init_normal;
    precision_type init_range;
	precision_type init_forget;
	bool norm_clipping;
	bool gradient_check;
	bool restart_states;
	
	bool use_feed_input;
	
    int num_noise_samples;

    bool use_momentum;
    precision_type initial_momentum;
    precision_type final_momentum;

    precision_type L2_reg;
	precision_type norm_threshold;

    bool normalization;
    precision_type normalization_init;

    int num_threads;
  
    bool share_embeddings;
	string training_sent_file;
	string validation_sent_file;
	string testing_sent_file;
	string load_encoder_file;
	string load_decoder_file;
	bool reverse_input; //Reverse input in while training the seq2seq LSTM
	double dropout_probability; 
	
	int max_epoch;

};

} // namespace nplm

