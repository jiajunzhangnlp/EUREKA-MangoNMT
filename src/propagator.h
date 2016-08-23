#ifndef NETWORK_H
#define NETWORK_H


#include "neuralClasses.h"
#include "util.h"
#include "graphClasses.h"
#include "SoftmaxLoss.h"

namespace nplm
{
	template <class input_node_type, class feed_input_node_type, class input_model_type, class feed_input_model_type>
	class propagator {
	public:
	    int minibatch_size;

	    model *encoder_plstm, *encoder_plstm_reverse, *decoder_plstm; //left_to_right, right_to_left encoder models and decoder model are the core components
		vector<LSTM_node<input_node_type,feed_input_node_type> > encoder_lstm_nodes; //encoder_lstm_nodes to calculate the presentation for each source position
		vector<LSTM_node<input_node_type,feed_input_node_type> > encoder_lstm_nodes_reverse; //calculate the presentation for each source position in reverse order
		vector<LSTM_node<input_node_type,feed_input_node_type> > decoder_lstm_nodes; //calculate the presentation for each target position
		vector<Attention_node<input_node_type,feed_input_node_type> > attention_nodes; //calculate the attention presentation for each target position
		vector<input_node_type > encoder_input_nodes, encoder_input_nodes_reverse, decoder_input_nodes; //input nodes for each source and target position
		vector<feed_input_node_type> feed_input_nodes; //feed input nodes for each target position
		Node<Output_word_embeddings> output_layer_node; //output layer nodes for each target position
		Matrix<precision_type,Dynamic,Dynamic> scores; 
		Matrix<precision_type,Dynamic,Dynamic> minibatch_weights;
		Matrix<precision_type,Dynamic,Dynamic> d_Err_t_d_output;
		Matrix<int,Dynamic,Dynamic> minibatch_samples;
		Matrix<int,Dynamic,Dynamic> minibatch_samples_no_negative;
		Matrix<precision_type,Dynamic,Dynamic> probs;	
		int num_hidden;
		precision_type fixed_partition_function; 
		boost::random::uniform_real_distribution<> unif_real;
		vector<Output_loss_node> losses; //loss nodes for each target position
		vector<Dropout_layer> output_dropout_layers; //output dropout
		SoftmaxNCELoss nce_loss;

	public:
		//Now we just allow at most 105 input words
	    propagator() : minibatch_size(0), 
					encoder_plstm(0), 
					encoder_plstm_reverse(0), 
					decoder_plstm(0),
					encoder_lstm_nodes(105,LSTM_node<input_node_type,feed_input_node_type>()),
					encoder_lstm_nodes_reverse(105,LSTM_node<input_node_type,feed_input_node_type>()),
					attention_nodes(105,Attention_node<input_node_type,feed_input_node_type>()),
					decoder_lstm_nodes(105,LSTM_node<input_node_type,feed_input_node_type>()),
					encoder_input_nodes(105,input_node_type()),
					encoder_input_nodes_reverse(105,input_node_type()),
					decoder_input_nodes(105,input_node_type()),
					feed_input_nodes(105,feed_input_node_type()),
					num_hidden(0), 
					fixed_partition_function(0), 
					losses(vector<Output_loss_node>(105,Output_loss_node())),
					unif_real(0.0,1.0),
					output_dropout_layers(vector<Dropout_layer>()),
					nce_loss(){ }

		//using the encoder and decoder models to initialize the encoding and decoding nodes
	    propagator (model &encoder_lstm, 
					model &encoder_lstm_reverse,
					model &decoder_lstm,
					int minibatch_size)
	      : encoder_plstm(&encoder_lstm),
	        encoder_plstm_reverse(&encoder_lstm_reverse),
			decoder_plstm(&decoder_lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&decoder_lstm.output_layer,minibatch_size),
			encoder_lstm_nodes(vector<LSTM_node<input_node_type,feed_input_node_type> >(105,LSTM_node<input_node_type,feed_input_node_type>(encoder_lstm,minibatch_size))),
			encoder_lstm_nodes_reverse(vector<LSTM_node<input_node_type,feed_input_node_type> >(105,LSTM_node<input_node_type,feed_input_node_type>(encoder_lstm_reverse,minibatch_size))),
			attention_nodes(vector<Attention_node<input_node_type,feed_input_node_type> >(105,Attention_node<input_node_type,feed_input_node_type>(decoder_lstm,minibatch_size))),
			decoder_lstm_nodes(vector<LSTM_node<input_node_type,feed_input_node_type> >(105,LSTM_node<input_node_type,feed_input_node_type>(decoder_lstm,minibatch_size))),
			encoder_input_nodes(vector<input_node_type >(105,input_node_type (dynamic_cast<input_model_type&>(*(encoder_lstm.input)),minibatch_size))),
			encoder_input_nodes_reverse(vector<input_node_type >(105,input_node_type (dynamic_cast<input_model_type&>(*(encoder_lstm_reverse.input)),minibatch_size))),
			decoder_input_nodes(vector<input_node_type >(105,input_node_type (dynamic_cast<input_model_type&>(*(decoder_lstm.input)),minibatch_size))),
			losses(vector<Output_loss_node>(105,Output_loss_node())),
			unif_real(0.0,1.0),
			output_dropout_layers(vector<Dropout_layer>()),
			nce_loss()
			{
				if( decoder_lstm.feed_input_flag )
					this->feed_input_nodes = vector<feed_input_node_type >(105,feed_input_node_type (dynamic_cast<feed_input_model_type&>(*(decoder_lstm.feed_input)),minibatch_size));
				else
					this->feed_input_nodes = vector<feed_input_node_type >(105,feed_input_node_type ());
				resize(minibatch_size);
			}
		
		void resizeOutput(int minibatch_size){
		  output_layer_node.resize(minibatch_size);			
		}
		void resizeRest(int minibatch_size) {
	      //this->minibatch_size = minibatch_size;
		  //CURRENTLY, THE RESIZING IS WRONG FOR SOME OF THE MINIBATCHES
		  scores.resize(output_layer_node.param->n_outputs(),minibatch_size); 
  		  minibatch_weights.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  minibatch_samples.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  probs.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  d_Err_t_d_output.resize(output_layer_node.param->n_outputs(),minibatch_size);		  			
		}
		
		void resizeEncoderNodes(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_lstm_nodes[i].resize(minibatch_size);
  			  encoder_lstm_nodes_reverse[i].resize(minibatch_size);
  		  }			
		}
		void resizeDecoderNodes(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_lstm_nodes[i].resize(minibatch_size);
  			  attention_nodes[i].resize(minibatch_size);
  		  }			
		}
		
		void resizeLosses(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  losses[i].resize(output_layer_node.param->n_inputs(),minibatch_size);
  		  }			
		}
		void resizeEncoderInputs(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_input_nodes[i].resize(minibatch_size);
  			  encoder_input_nodes_reverse[i].resize(minibatch_size);
  			  encoder_lstm_nodes[i].set_input_node(encoder_input_nodes[i]);
  			  encoder_lstm_nodes_reverse[i].set_input_node(encoder_input_nodes_reverse[i]);

  		  }			
		}
		void resizeDecoderInputs(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_input_nodes[i].resize(minibatch_size);
  			  decoder_lstm_nodes[i].set_input_node(decoder_input_nodes[i]);
			  if( decoder_plstm->feed_input_flag ) {
				  decoder_lstm_nodes[i].set_feed_input_flag(true);
				  feed_input_nodes[i].resize(minibatch_size);
				  decoder_lstm_nodes[i].set_feed_input_node(feed_input_nodes[i]);
			  }
  		  }			
		}

		void resizeEncoderInputsDropout(int minibatch_size, precision_type dropout_probability){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_input_nodes[i].resizeDropout(minibatch_size, dropout_probability);
  			  encoder_input_nodes_reverse[i].resizeDropout(minibatch_size, dropout_probability);
  			  encoder_lstm_nodes[i].set_input_node(encoder_input_nodes[i]);
  			  encoder_lstm_nodes_reverse[i].set_input_node(encoder_input_nodes_reverse[i]);

  		  }			
		}
		void resizeDecoderInputsDropout(int minibatch_size, precision_type dropout_probability){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_input_nodes[i].resizeDropout(minibatch_size, dropout_probability);
  			  decoder_lstm_nodes[i].set_input_node(decoder_input_nodes[i]);
			  if( decoder_plstm->feed_input_flag ) {
				  decoder_lstm_nodes[i].set_feed_input_flag(true);
				  feed_input_nodes[i].resize(minibatch_size);
				  decoder_lstm_nodes[i].set_feed_input_node(feed_input_nodes[i]);
			  }
  		  }			
		}
				
	    void resize(int minibatch_size) {
			this->minibatch_size = minibatch_size;
			resizeOutput(minibatch_size);
			resizeEncoderNodes(minibatch_size);
			resizeDecoderNodes(minibatch_size);
			resizeEncoderInputs(minibatch_size);
			resizeDecoderInputs(minibatch_size);
			resizeLosses(minibatch_size);
			resizeRest(minibatch_size);
		}
		
		void resizeOutputDropoutLayers(int minibatch_size, precision_type dropout_probability) {
			this->output_dropout_layers = vector<Dropout_layer>(105,Dropout_layer(output_layer_node.param->n_inputs(),
																		minibatch_size,
																		1-dropout_probability));
		}
		
	    void resizeDropout(int minibatch_size, precision_type dropout_probability) {
			this->minibatch_size = minibatch_size;
			resizeOutput(minibatch_size);
			resizeEncoderNodes(minibatch_size);
			resizeDecoderNodes(minibatch_size);
			resizeEncoderInputsDropout(minibatch_size, dropout_probability);
			resizeDecoderInputsDropout(minibatch_size, dropout_probability);
			resizeLosses(minibatch_size);
			resizeRest(minibatch_size);
			resizeOutputDropoutLayers(minibatch_size, dropout_probability);
		}		

		
		//Resizing some of the NCE mibatch matrices
		//template <typename X> 
		//I should template multinomial<data_size_t>
		void resizeNCE(int num_noise_samples, precision_type fixed_partition_function, multinomial<data_size_t> &unigram){
			minibatch_weights.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples_no_negative.setZero(num_noise_samples+1,minibatch_size);
			scores.setZero(num_noise_samples+1,minibatch_size);
			probs.setZero(num_noise_samples+1,minibatch_size);
			//cerr<<"Size of scores is "<<scores.cols()<<" "<<scores.rows()<<endl;
			this->fixed_partition_function = fixed_partition_function;
			this->nce_loss.set_unigram(&unigram);
			d_Err_t_d_output.resize(num_noise_samples+1,minibatch_size);	
		}
	    void resize() { resize(minibatch_size); }
		
		//Both the input and the output sentences are columns. Even it is a minibatch of sentences, each sentence is a column
		/*************************************************************************************
		Decoder Forward propagation
		Input: output_data -- gold output data used as the input to LSTM
			   const_current_c -- initial cell state for decoder LSTM sequence
			   const_current_h -- initial hidden state for decoder LSTM sequence
			   const_source_ctx -- source-side hidden states used for attention model
			   sequence_cont_indices -- continuation vector for each sentence in a minibatch
			   sequence_token_num -- token number for each sentence in a minibatch 
			   max_input_sent_len -- maximum sentence length in a minibatch
		Output: attention hidden states
		**************************************************************************************/
	    template <typename DerivedOutput, typename DerivedS>
	    void fPropDecoder(const MatrixBase<DerivedOutput> &output_data,
				const Matrix<precision_type,Dynamic,Dynamic> &const_current_c,
				const Matrix<precision_type,Dynamic,Dynamic> &const_current_h,
				const vector<Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				const vector<int> &sequence_token_num,
				const unsigned int &max_input_sent_len)
	    {

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int output_sent_len = output_data.rows();
			int current_minibatch_size = output_data.cols();
			//Going over the output sentence to generate the hidden states
			for (int i=0; i<output_sent_len; i++){
				if (i==0) {
					//copy const_current_h and const_current_c to h_t_minus_one and c_t_minus_one
					decoder_lstm_nodes[i].filterStatesAndErrors(const_current_h,
																const_current_c,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));			
																
					//if we apply feed input, there are two inputs: one is word output_data.row(i) and the other is last attention hidden state
					if( decoder_plstm->feed_input_flag ) {
						int decoder_num_hidden = decoder_lstm_nodes[0].h_t.rows();
						Matrix<precision_type,Dynamic,Dynamic> init_feed_input_h;
						init_feed_input_h.setZero(decoder_num_hidden,minibatch_size);
						decoder_lstm_nodes[i].fPropFeedInput(output_data.row(i), init_feed_input_h);
					}
					else {
						decoder_lstm_nodes[i].fProp(output_data.row(i));
					}
				} else {
					//copy h_t and c_t in previous node to h_t_minus_one and c_t_minus_one
					decoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[i-1].h_t,
																decoder_lstm_nodes[i-1].c_t,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));					
																
					if( decoder_plstm->feed_input_flag ) //attention_nodes[i-1].h_new_t is directly linked to the next decoder LSTM node
						decoder_lstm_nodes[i].fPropFeedInput(output_data.row(i), attention_nodes[i-1].h_new_t);
					else
						decoder_lstm_nodes[i].fProp(output_data.row(i));
				}
				//apply attention model to generate the attention hidden state using decoder LSTM output and the source-side hidden states
				attention_nodes[i].fProp(decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
			}			
	    }
		
		//Both the input and the output sentences are columns. Even it is a minibatch of sentences, each sentence is a column
		//Same as fPropDecoderDropout except that fPropDecoderDropout applies dropout for input data
	    template <typename DerivedOutput, typename DerivedS, typename Engine>
	    void fPropDecoderDropout(const MatrixBase<DerivedOutput> &output_data,
				const Matrix<precision_type,Dynamic,Dynamic> &const_current_c,
				const Matrix<precision_type,Dynamic,Dynamic> &const_current_h,
				const vector<Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				const vector<int> &sequence_token_num,
				const unsigned int &max_input_sent_len,
				Engine &eng)
	    {
			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int output_sent_len = output_data.rows();
			int current_minibatch_size = output_data.cols();

			//Going over the output sentence to generate the hidden states
			for (int i=0; i<output_sent_len; i++){
				if (i==0) {
					decoder_lstm_nodes[i].filterStatesAndErrors(const_current_h,
																const_current_c,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));			
																
					if( decoder_plstm->feed_input_flag ) {
						int decoder_num_hidden = decoder_lstm_nodes[0].h_t.rows();
						Matrix<precision_type,Dynamic,Dynamic> init_feed_input_h;
						init_feed_input_h.setZero(decoder_num_hidden,minibatch_size);
						decoder_lstm_nodes[i].fPropDropoutFeedInput(output_data.row(i), init_feed_input_h, eng);
					}
					else {
						decoder_lstm_nodes[i].fPropDropout(output_data.row(i), eng);
					}
				} else {
					decoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[i-1].h_t,
																decoder_lstm_nodes[i-1].c_t,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));						
																
					if( decoder_plstm->feed_input_flag )
						decoder_lstm_nodes[i].fPropDropoutFeedInput(output_data.row(i), attention_nodes[i-1].h_new_t, eng);
					else
						decoder_lstm_nodes[i].fPropDropout(output_data.row(i), eng);
				}
				attention_nodes[i].fProp(decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
			}			

	    }
				
		/****************************************************************************************
		 Encoder left-to-right forward propagation
		 Input: input_data -- source-side input sentences
		        const_current_c -- initial cell state for encoder LSTM node sequence
			    const_current_h -- initial hidden state for encoder LSTM node sequence
			    sequence_cont_indices -- sentence continuation vector for a minibatch
		 Output: hidden state sequence	
		*****************************************************************************************/
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropEncoder(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			int current_minibatch_size = input_data.cols();

			for (int i=0; i<sent_len; i++){
				if (i==0) {
					encoder_lstm_nodes[i].filterStatesAndErrors(current_h,
																current_c,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));	
					encoder_lstm_nodes[i].fProp(input_data.row(i)); //ith words as input
				} else {
					encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i-1].h_t,
																encoder_lstm_nodes[i-1].c_t,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i-1)); 
					encoder_lstm_nodes[i].fProp(input_data.row(i));
				}
			}
	    }

		/****************************************************************************************
		 Encoder right-to-left forward propagation
		 Input: input_data -- source-side input sentences in reverse order
		        const_current_c -- initial cell state for encoder LSTM node sequence
			    const_current_h -- initial hidden state for encoder LSTM node sequence
			    sequence_cont_indices -- sentence continuation vector for a minibatch
		 Output: hidden state sequence	
		*****************************************************************************************/
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropEncoderReverse(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			int current_minibatch_size = input_data.cols();

			for (int i=0; i<sent_len; i++){
				if (i==0) {
					encoder_lstm_nodes_reverse[i].filterStatesAndErrors(current_h,
																current_c,
																encoder_lstm_nodes_reverse[i].h_t_minus_one,
																encoder_lstm_nodes_reverse[i].c_t_minus_one,
																sequence_cont_indices.row(i));	
					encoder_lstm_nodes_reverse[i].fProp(input_data.row(i));
				} else {
					encoder_lstm_nodes_reverse[i].filterStatesAndErrors(encoder_lstm_nodes_reverse[i-1].h_t,
																encoder_lstm_nodes_reverse[i-1].c_t,
																encoder_lstm_nodes_reverse[i].h_t_minus_one,
																encoder_lstm_nodes_reverse[i].c_t_minus_one,
																sequence_cont_indices.row(i-1)); 
					encoder_lstm_nodes_reverse[i].fProp(input_data.row(i));

				}
			}
	    }

		//Same as fPropEncoder except that fPropEncoderDropout applies dropout for input data
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS, typename Engine>
	    void fPropEncoderDropout(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				Engine &eng)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			int current_minibatch_size = input_data.cols();
			
			//Going over the input sentence to generate the hidden states
			for (int i=0; i<sent_len; i++){
				if (i==0) {
					encoder_lstm_nodes[i].filterStatesAndErrors(current_h,
																current_c,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));	
					encoder_lstm_nodes[i].fPropDropout(input_data.row(i), eng);
				} else {
					encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i-1].h_t,
																encoder_lstm_nodes[i-1].c_t,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i-1));
					encoder_lstm_nodes[i].fPropDropout(input_data.row(i), eng);
				}
			}
	    }


		//Same as fPropEncoderReverse except that fPropEncoderDropoutReverse applies dropout for input data
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS, typename Engine>
	    void fPropEncoderDropoutReverse(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				Engine &eng)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			int current_minibatch_size = input_data.cols();
			
			//Going over the input sentence to generate the hidden states
			for (int i=0; i<sent_len; i++){
				if (i==0) {
					encoder_lstm_nodes_reverse[i].filterStatesAndErrors(current_h,
																current_c,
																encoder_lstm_nodes_reverse[i].h_t_minus_one,
																encoder_lstm_nodes_reverse[i].c_t_minus_one,
																sequence_cont_indices.row(i));	
					encoder_lstm_nodes_reverse[i].fPropDropout(input_data.row(i), eng);
				} else {
					encoder_lstm_nodes_reverse[i].filterStatesAndErrors(encoder_lstm_nodes_reverse[i-1].h_t,
																encoder_lstm_nodes_reverse[i-1].c_t,
																encoder_lstm_nodes_reverse[i].h_t_minus_one,
																encoder_lstm_nodes_reverse[i].c_t_minus_one,
																sequence_cont_indices.row(i-1));
					encoder_lstm_nodes_reverse[i].fPropDropout(input_data.row(i), eng);
				}
			}
	    }
				
		template <typename DerivedH> 
		void getHiddenStates(const MatrixBase<DerivedH> &const_hidden_states,
							const int max_sent_len,
							const bool for_encoder,
							const int sent_index) const {
			//if for_encoder is 1, then you will print the hidden states for left-to-right encoder, 2 for right-to-left encoder,  else decoder
			UNCONST(DerivedH, const_hidden_states, hidden_states);
			int num_hidden = decoder_lstm_nodes[0].h_t.rows();
			for(int i=0; i<max_sent_len; i++){
				if (for_encoder == 1) {
					hidden_states.col(i) = encoder_lstm_nodes[i].h_t.col(sent_index);
				}
				else if (for_encoder == 2) {
					hidden_states.col(i) = encoder_lstm_nodes_reverse[i].h_t.col(sent_index);
				} else {
					hidden_states.col(i) = decoder_lstm_nodes[i].h_t.col(sent_index);
				}
			}
		}
		
		//get internal states for encoder and decoder LSTM node sequence
	    template<typename DerivedO, typename DerivedF, typename DerivedI, typename DerivedH, typename DerivedC>		
	    void getInternals(const MatrixBase<DerivedH> &const_get_h_t,
						const MatrixBase<DerivedC>   &const_get_c_t,
						const MatrixBase<DerivedF>   &const_get_f_t,
						const MatrixBase<DerivedI>   &const_get_i_t,
						const MatrixBase<DerivedO>   &const_get_o_t,
						const int max_sent_len,
						const bool for_encoder,
						const int sent_index){
			UNCONST(DerivedH, const_get_h_t, get_h_t);
			UNCONST(DerivedH, const_get_c_t, get_c_t);
			UNCONST(DerivedH, const_get_f_t, get_f_t);
			UNCONST(DerivedH, const_get_i_t, get_i_t);
			UNCONST(DerivedH, const_get_o_t, get_o_t);			
			for (int i=0; i<max_sent_len; i++){
				if (for_encoder == 1){
					encoder_lstm_nodes[i].getInternals(get_h_t.col(i),
													get_c_t.col(i),
													get_f_t.col(i),
													get_i_t.col(i),
													get_o_t.col(i),
													sent_index);
				}
				else if (for_encoder == 2){
					encoder_lstm_nodes_reverse[i].getInternals(get_h_t.col(i),
													get_c_t.col(i),
													get_f_t.col(i),
													get_i_t.col(i),
													get_o_t.col(i),
													sent_index);
				} else {
					decoder_lstm_nodes[i].getInternals(get_h_t.col(i),
													get_c_t.col(i),
													get_f_t.col(i),
													get_i_t.col(i),
													get_o_t.col(i),
													sent_index);
				}
			}
	    }

					
		//currently only generate one output at a time
		/*********************************************************************************
		 * sequence-to-sequence prediction with greedy algorithm in decoder
		 * Input: input_data -- source-side input sentences
		 *        decoder_input_vocab -- vocabulary for decoder input
		 *        decoder_output_vocab -- vocabulary for decoder output (decoder output at current time will be decoder input at next time)
		 *        const_current_c -- initial cell state for decoder LSTM node sequence
		 *        const_current_h -- initial hidden state for decoder LSTM node sequence
		 *        const_source_ctx -- source-side hidden state sequence
		 *        sequence_token_num -- token number for each sentence in a minibatch
		 *        output_start_symbol -- <s>
		 *        output_end_symbol -- </s>
		 * Output:
		 *        predicted_sequence -- greedy output
		*********************************************************************************/
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateGreedyOutput(const MatrixBase<DerivedInput> &input_data,
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const vector<Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
				const vector<int> &sequence_token_num,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol) {
					int current_minibatch_size = input_data.cols();
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(101,current_minibatch_size); // I can produce at most 100 output symbols
					predicted_output.row(0).fill(output_start_symbol); //assume the first output words are output_start_symbol
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	

					vector<int> live_words (current_minibatch_size,1); //whether the generation is completed for each output sentence in minibatch
					int remaining_live_words = current_minibatch_size; //how many sentences are not generated completely
					
				for (int i=0; i<100 && remaining_live_words > 0; i++){
					if (i==0) {
						//copy current_h and current_c to h_t_minusone and c_t_minusone
						//obtain the decoder LSTM hidden state first
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,
											current_c);
						if( decoder_plstm->feed_input_flag ) {
							int decoder_num_hidden = decoder_lstm_nodes[0].h_t.rows();
							Matrix<precision_type,Dynamic,Dynamic> init_feed_input_h;
							init_feed_input_h.setZero(decoder_num_hidden,minibatch_size);
							decoder_lstm_nodes[i].fPropFeedInput(predicted_output.row(i), init_feed_input_h);
						}
						else {
							decoder_lstm_nodes[i].fProp(predicted_output.row(i));
						}
					} else {
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,
											decoder_lstm_nodes[i-1].c_t);
						if( decoder_plstm->feed_input_flag )
							decoder_lstm_nodes[i].fPropFeedInput(predicted_output.row(i), attention_nodes[i-1].h_new_t);//,
						else
							decoder_lstm_nodes[i].fProp(predicted_output.row(i));
					}
					//generate attention hidden state with source-side hidden states and current decoder hidden state
					attention_nodes[i].fProp(decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, input_data.rows());
					//calculate output scores with attention hidden states
					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));
	
					//calcuate the propbabilities using softmax
			        precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	

					for( int index=0; index<live_words.size(); index++ ){
						if( live_words[index] == 1 ){ //if current output sentence is not generated completely, find the output with maximum probability
							Matrix<precision_type,1,Dynamic>::Index max_index;
							probs.col(index).maxCoeff(&max_index);
							if (max_index == output_end_symbol){
								live_words[index] = -1;
								remaining_live_words--;
							} 
							//Because the decoder output vocabulary is different from
							//the decoder input vocabulary, the max index is from the 
							//decoder output vocabulary. However, when we populate 
							//predicted_output, we have to use the decoder input vocabulary.
							predicted_sequence[index].push_back(max_index);
							predicted_output(i+1,index) = decoder_input_vocab.lookup_word(decoder_output_vocab.get_word(max_index)); //prepare the input for next LSTM node
						} else {
							predicted_output(i+1,index) = 0; //it's best to make it 0
						}
					}
				}
		}

		//currently only process one input sentence at a time
		/*********************************************************************************
		 * sequence-to-sequence prediction with beam search algorithm in decoder
		 * Input: input_data -- source-side input sentences, currently we deal with one sentence each time, that is minibatch size is 1
		 *        decoder_input_vocab -- vocabulary for decoder input
		 *        decoder_output_vocab -- vocabulary for decoder output (decoder output at current time will be decoder input at next time)
		 *        const_current_c -- initial cell state for decoder LSTM node sequence
		 *        const_current_h -- initial hidden state for decoder LSTM node sequence
		 *        const_source_ctx -- source-side hidden state sequence
		 *        sequence_token_num -- token number for each sentence in a minibatch
		 *        output_start_symbol -- <s>
		 *        output_end_symbol -- </s>
		 * Output:
		 *        final_k_best_seq_list -- beam-search output
		*********************************************************************************/
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void beamDecoding(const MatrixBase<DerivedInput> &input_data,
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const vector<Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
				const vector<int> &sequence_token_num,
				vector<k_best_seq_item> &final_k_best_seq_list,
				const int output_start_symbol,
				const int output_end_symbol,
				const int beam_size) {
					//Note that all columns denote k-best rather than minibatch
					int k = beam_size;
					int current_beam_size, previous_beam_size;
					current_beam_size = previous_beam_size = k;
					int current_minibatch_size = 1;
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(1,1); //For now, we are processing one sentence at a time. 
												  //In the beginning, there is only one symbol,<s>. 
												  //After that, we will have k items in predicted output
					predicted_output.row(0).fill(output_start_symbol);

				//First generate k items to populate the beam
				//At first, generate the decoder hidden state
				decoder_lstm_nodes[0].copyToHiddenStates(const_current_h.leftCols(1),
									const_current_c.leftCols(1));
				if( decoder_plstm->feed_input_flag ) {
					int decoder_num_hidden = decoder_lstm_nodes[0].h_t.rows();
					Matrix<precision_type,Dynamic,Dynamic> init_feed_input_h;
					init_feed_input_h.setZero(decoder_num_hidden,minibatch_size);
					decoder_lstm_nodes[0].fPropFeedInput(predicted_output.row(0), init_feed_input_h);
				}
				else
					decoder_lstm_nodes[0].fProp(predicted_output.row(0));
				//then generate the attention hidden state
				attention_nodes[0].fProp(decoder_lstm_nodes[0].h_t, const_source_ctx, sequence_token_num, input_data.rows());
				//Get the k-best items from the list first
				//Note that if vocab size < k, then the initial k-best list will have less than k items.
				//there are three objects in struct beam_item: value -- prob of current hypo; row -- output word location; col -- which previous hypo this hypo comes from
				vector<beam_item> initial_k_best_list;
				output_layer_node.param->fProp(attention_nodes[0].h_new_t.leftCols(1), 
									scores.leftCols(1));
		        SoftmaxLogLoss().computeProbs(scores.leftCols(1), 
		                   predicted_output.row(0), 
		                   probs);					
				//there are two objects in struct k_best_seq_item: value -- prob of current sequence hypo; seq -- current sequence hypo
				vector<k_best_seq_item> k_best_seq_list;		   					
				getKBest(probs.leftCols(1), initial_k_best_list, k_best_seq_list, k); //get k-best according to props

				//Initial k-best list might have less than k items.
				current_beam_size = initial_k_best_list.size();
				//Now populate the k-best sequences with the initial k best list
				assert(initial_k_best_list.size() <= k);
				for (int i=0; i<initial_k_best_list.size(); i++)	{
					k_best_seq_item seq_item;
					seq_item.seq.push_back(initial_k_best_list.at(i).row); //Row indicates the word index in the probability matrix
					seq_item.value = initial_k_best_list.at(i).value;
					k_best_seq_list.push_back(seq_item);
				}

				//Now create a new predicted output from the k_best list and also transfer the hidden states
				//For the 1st position, all the hidden states will come from the same hidden state of just seeing <s>
				vector<int> k_best_state_copy_indices = vector<int>(initial_k_best_list.size(),0);
				predicted_output.resize(101,k); // I can produce at most 100 output symbols
				//prepare the input for next LSTM node
				for (int i=0; i<initial_k_best_list.size();  i++){
					predicted_output(1,i) = initial_k_best_list.at(i).row;
					string string_output_word = decoder_output_vocab.get_word(initial_k_best_list.at(i).row);
					predicted_output(1,i) = decoder_input_vocab.lookup_word(string_output_word);
				}

				for (int i=1; i<100 ; i++){
					//first get the decoder LSTM hidden state
					//k_best_state_copy_indices indicates which previous hypo does current hypo extended from
					decoder_lstm_nodes[i].copyKBestHiddenStates(decoder_lstm_nodes[i-1].h_t,
										  decoder_lstm_nodes[i-1].c_t,
										  decoder_lstm_nodes[i].h_t_minus_one,
										  decoder_lstm_nodes[i].c_t_minus_one,
										  k_best_state_copy_indices);
					if( decoder_plstm->feed_input_flag ) {
						Matrix<precision_type,Dynamic,Dynamic> current_attention_h;
						current_attention_h.setZero(attention_nodes[0].h_new_t.rows(), current_beam_size);
						attention_nodes[i].copyKBestHiddenStates(attention_nodes[i-1].h_new_t, current_attention_h, k_best_state_copy_indices);
						decoder_lstm_nodes[i].fPropFeedInput(predicted_output.row(i).leftCols(current_beam_size), current_attention_h);
					}
					else {
						decoder_lstm_nodes[i].fProp(predicted_output.row(i).leftCols(current_beam_size));
					}

					//obtain the attention hidden state
					vector<int> minibatch_token_num(current_beam_size, sequence_token_num[0]); //all the hypos share the same source sentence
					vector< Matrix<precision_type,Dynamic,Dynamic> > current_source_ctx;
					current_source_ctx.resize(const_source_ctx.size());
					for( int src_id = 0; src_id < const_source_ctx.size(); src_id++ ) {
						current_source_ctx[src_id].resize(const_source_ctx[0].rows(), current_beam_size);
						for( int sen_id = 0; sen_id < current_beam_size; sen_id++ ) {
							current_source_ctx[src_id].col(sen_id) = const_source_ctx[src_id].col(0);
						}
					}
					attention_nodes[i].fProp(decoder_lstm_nodes[i].h_t, current_source_ctx, minibatch_token_num, input_data.rows());

					//calculate probabilities
					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_beam_size), 
										scores.leftCols(current_beam_size));
			        start_timer(5);
			        SoftmaxLogLoss().computeProbs(scores.leftCols(current_beam_size), 
			                   predicted_output.row(i), 
			                   probs);	
					vector<beam_item> k_best_list;

					//Extracting at most 2k best items because we could have k items with end symbols
					//Now that we have the k-best list, we add to the k_best_seq_list and remove the previous items.					
					getKBest(probs.leftCols(current_beam_size), k_best_list, k_best_seq_list, 2*k);
					assert(k_best_list.size() <= 2*k);
					int new_k = 0;
					k_best_state_copy_indices.clear();
					for (int item_index=0; new_k<k && item_index < k_best_list.size() ; item_index++){
						k_best_seq_item seq_item;
						int prev_k_best_seq_item_index = k_best_list.at(item_index).col;
						int word_index = k_best_list.at(item_index).row;
						
						seq_item.seq = k_best_seq_list.at(prev_k_best_seq_item_index).seq;
						
						seq_item.seq.push_back(word_index);
						seq_item.value = k_best_list.at(item_index).value;
						if (word_index != output_end_symbol){
							k_best_seq_list.push_back(seq_item);
							//The hidden state to be transmitted to the next LSTM block is 
							//the one with the index of the previous k_best seq item 
							k_best_state_copy_indices.push_back(prev_k_best_seq_item_index);
							
							//We need to convert the output word index into the input word index
							string string_output_word = decoder_output_vocab.get_word(word_index);
							//cerr<<"string_output_word "<<string_output_word<<endl;
							predicted_output(i+1,new_k) = decoder_input_vocab.lookup_word(string_output_word);							
							//predicted_output(i+1,new_k) = word_index;
							
							new_k++;
						} else {
							final_k_best_seq_list.push_back(seq_item);
						}
					}

					//The k_best_seq_list has expanded now and the previou previous_beam_size-items should be deleted
					previous_beam_size = current_beam_size;
					current_beam_size = new_k;
					assert (k_best_seq_list.size() <= 2*k);
					k_best_seq_list.erase(k_best_seq_list.begin(), k_best_seq_list.begin()+previous_beam_size);
				}
				//Now adding the k_best_seq list to the final list
				for (int item_index=0; item_index<k_best_seq_list.size(); item_index++){
					final_k_best_seq_list.push_back(k_best_seq_list.at(item_index));
				}
				//First soring the k_best_list
				//Getting the average symbol probability
				std::make_heap(final_k_best_seq_list.begin(), final_k_best_seq_list.end(), comparator<k_best_seq_item>());
				std::sort_heap(final_k_best_seq_list.begin(), final_k_best_seq_list.end(), comparator<k_best_seq_item>());
		}

		//currently only generate one output at a time
		//It is similar to generateGreedyOutput and the difference lies in that generateStochasticOutput predict output words stochastically rather than greedy or beam-search
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateStochasticOutput(const MatrixBase<DerivedInput> &input_data,
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const vector< Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
				const vector<int> &minibatch_input_token_num,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol,
				boost::random::mt19937 &eng) {
					int current_minibatch_size = input_data.cols();
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(101,current_minibatch_size); // I can produce at most 100 output symbols
					predicted_output.row(0).fill(output_start_symbol);
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	
					vector<int> live_words (current_minibatch_size,1);
					int remaining_live_words = current_minibatch_size;
					
				for (int i=0; i<101 && remaining_live_words > 0; i++){

					if (i==0) {
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,
											current_c);
						if( decoder_plstm->feed_input_flag ) {
							int decoder_num_hidden = decoder_lstm_nodes[0].h_t.rows();
							Matrix<precision_type,Dynamic,Dynamic> init_feed_input_h;
							init_feed_input_h.setZero(decoder_num_hidden,minibatch_size);
							decoder_lstm_nodes[i].fPropFeedInput(predicted_output.row(i), init_feed_input_h);
						}
						else {
							decoder_lstm_nodes[i].fProp(predicted_output.row(i));
						}
					} else {
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,
											decoder_lstm_nodes[i-1].c_t);
						if( decoder_plstm->feed_input_flag )
							decoder_lstm_nodes[i].fPropFeedInput(predicted_output.row(i), attention_nodes[i-1].h_new_t);
						else
							decoder_lstm_nodes[i].fProp(predicted_output.row(i));
					}
					attention_nodes[i].fProp(decoder_lstm_nodes[i].h_t, const_source_ctx, minibatch_input_token_num, input_data.rows());
					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));

			        precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	

					for (int index=0; index<live_words.size(); index++){
						if (live_words[index] == 1){
							precision_type rand_value = unif_real(eng);
							int rand_index =0;
							precision_type cumul =0;
							for (int vocab_index=0; vocab_index<probs.col(index).rows(); vocab_index++){
								cumul += exp(probs(vocab_index,index));
								if (cumul >= rand_value){
									rand_index = vocab_index;
									break;
								}
								
							}
							if (rand_index == output_end_symbol){
								live_words[index] = -1;
								remaining_live_words--;
							} 
							predicted_sequence[index].push_back(rand_index);
							predicted_output(i+1,index) = decoder_input_vocab.lookup_word(decoder_output_vocab.get_word(rand_index));
						} else {
							predicted_output(i+1,index) = 0;
						}
					}
				}
		}
		
		template<typename Derived>
		void printHiddenStates(const MatrixBase<Derived> &matrix, const string &type){
			for (int i=0; i<matrix.cols(); i++){
				cerr<<"state "<<type<<":"<<i<<" has norm "<<matrix.col(i).norm()<<endl;
			}
		}
			
		//Computing losses separately. Makes more sense because some LSTM units might not output units but will be receiving 
		//losses from the next layer
		/*********************************************************************************
		 * Computing loss according to the attention hidden states and the gold output
		 * Input: output -- target-side gold output
		 *        gradient_check -- flag indicating whether do gradient check
		 *        norm_clipping -- flag indicating which strategy is employed to avoid gradient from exploding or vanishing
		 *        loss_function -- variable indicating which loss function is used
		 *        unigram -- the unigram for training data used for NCE optimization
		 *        num_noise_samples -- the number of nise samples we apply in NCE
		 * Output:
		 * 		  log_likelihodd -- the loss
		*********************************************************************************/
	    template <typename DerivedOut, typename data_type> 
		void computeLosses(const MatrixBase<DerivedOut> &output,
			 precision_type &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 loss_function_type loss_function,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng) {
	 			int current_minibatch_size = output.cols();
	 			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);

	 			int sent_len = output.rows(); 
	 			for (int i=sent_len-1; i>=0; i--) {
					precision_type minibatch_log_likelihood;
	 				if (loss_function == LogLoss) {
	 					//First doing fProp for the output layer
	 					//The number of columns in scores will be the current minibatch size
	 					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), scores);
				
						//calculate the probs using softmax
	 			        start_timer(5);
	 			        SoftmaxLogLoss().fProp(scores, 
	 			                   output.row(i), 
	 			                   probs, 
	 			                   minibatch_log_likelihood);
	 			        stop_timer(5);
						//update the log likelihood
	 			        log_likelihood += minibatch_log_likelihood;

	 			        ///// Backward propagation
						//softmax backpropagation
	 			        start_timer(6);
	 	   		        SoftmaxLogLoss().bProp(output.row(i), 
	 	   		                   probs.leftCols(current_minibatch_size), 
	 	   		                   d_Err_t_d_output);
	 			        stop_timer(6);
				
	 					//Now computing the derivative of the output layer
	 					//The number of colums in output_layer_node.bProp_matrix will be the current minibatch size
	 	   		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
										losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
						//update the output word embeddings
	 	   		        output_layer_node.param->updateGradient(attention_nodes[i].h_new_t.leftCols(current_minibatch_size),
	 	   						       d_Err_t_d_output.leftCols(current_minibatch_size));									   	 		   
	 				} else if (loss_function == NCELoss){
						//generate specific noise samples for each word in minibatch
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						//prepare the positive samples in the first row
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);

						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}

						scores.setZero();
						//output layer forward propagation
						output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
														minibatch_samples_no_negative.leftCols(current_minibatch_size),
														scores);
						//nce loss layer forward propagation
						nce_loss.fProp(scores, 
	       			 				  minibatch_samples,
	       						   	  probs, 
		   						  	  minibatch_log_likelihood,
									  this->fixed_partition_function);
						//update likelihood
						log_likelihood += minibatch_log_likelihood;
							
						//nce loss layer backpropagation
						nce_loss.bProp(probs.leftCols(current_minibatch_size),
										d_Err_t_d_output);
						//output layer backpropagation
			 	   		output_layer_node.param->bProp(minibatch_samples_no_negative.leftCols(current_minibatch_size),
			 	   									d_Err_t_d_output.leftCols(current_minibatch_size),
													losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
						//cerr<<attention_nodes[i].h_new_t.leftCols(current_minibatch_size)<<endl;
						//cerr<<minibatch_samples_no_negative.leftCols(current_minibatch_size)<<endl;
						//cerr<<d_Err_t_d_output.leftCols(current_minibatch_size)<<endl;
						//update output word embeddings
			 	   		output_layer_node.param->updateGradient(attention_nodes[i].h_new_t.leftCols(current_minibatch_size),
													     minibatch_samples_no_negative.leftCols(current_minibatch_size),
													     d_Err_t_d_output.leftCols(current_minibatch_size));
	 				}
	 			}
		}
		
		//generate specific noise samples for each word in minibatch
		template <typename Derived, typename Engine> 
		void generateSamples(MatrixBase<Derived> const &minibatch, multinomial<data_size_t> &unigram, Engine &eng){	
			
			UNCONST(Derived, minibatch, my_minibatch);

			//if the words in the minibatch share the same noise samples
			#ifdef SHARE_SAMPLES
			for (int row=0; row<my_minibatch.rows(); row++){
				int sample = unigram.sample(eng);
				my_minibatch.row(row).fill(sample);
			}	
			#else 		
			for (int row=0; row<my_minibatch.rows(); row++){
				for (int col=0; col<my_minibatch.cols(); col++){
					my_minibatch(row,col) = unigram.sample(eng);
				}
			}
			#endif	
		}

		//Computing losses separately. Makes more sense because some LSTM units might not output units but will be receiving 
		//losses from the next layer
		//Same as computeLosses except that computeLossesDropout applies dropout for output layer
	    template <typename DerivedOut, typename data_type> 
		void computeLossesDropout(const MatrixBase<DerivedOut> &output,
			 precision_type &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 loss_function_type loss_function,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng) {
	 			int current_minibatch_size = output.cols();
	 			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in and LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
	 			int sent_len = output.rows(); 
			
	 			for (int i=sent_len-1; i>=0; i--) {
					precision_type minibatch_log_likelihood;
	 				if (loss_function == LogLoss) {
						//Applying dropout to the output layer
						output_dropout_layers[i].fProp(attention_nodes[i].h_new_t,rng);
	 					//First doing fProp for the output layer
	 					//The number of columns in scores will be the current minibatch size
	 					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), scores);
	 			        
						//calculate the probs with attention hidden state
	 			        start_timer(5);
	 			        SoftmaxLogLoss().fProp(scores, 
	 			                   output.row(i), 
	 			                   probs, 
	 			                   minibatch_log_likelihood);
	 			        stop_timer(5);
	 			        log_likelihood += minibatch_log_likelihood;

	 			        ///// Backward propagation
	 			        start_timer(6);
	 	   		        SoftmaxLogLoss().bProp(output.row(i), 
	 	   		                   probs.leftCols(current_minibatch_size), 
	 	   		                   d_Err_t_d_output);
	 			        stop_timer(6);

	 					//Now computing the derivative of the output layer
	 					//The number of colums in output_layer_node.bProp_matrix will be the current minibatch size
	 	   		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
										losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));

						//update output word embeddings
	 	   		        output_layer_node.param->updateGradient(attention_nodes[i].h_new_t.leftCols(current_minibatch_size),
	 	   						       d_Err_t_d_output.leftCols(current_minibatch_size));	
	 				} else if (loss_function == NCELoss){
						output_dropout_layers[i].fProp(attention_nodes[i].h_new_t,rng);
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);

						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}
						scores.setZero();
						output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
														minibatch_samples_no_negative.leftCols(current_minibatch_size),
														scores);
						nce_loss.fProp(scores, 
	       			 				  minibatch_samples,
	       						   	  probs, 
		   						  	  minibatch_log_likelihood,
									  this->fixed_partition_function);
						log_likelihood += minibatch_log_likelihood;
							
						//backpropagation
						nce_loss.bProp(probs.leftCols(current_minibatch_size),
										d_Err_t_d_output);
			 	   		output_layer_node.param->bProp(minibatch_samples_no_negative.leftCols(current_minibatch_size),
			 	   									d_Err_t_d_output.leftCols(current_minibatch_size),
													losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
			 	   		output_layer_node.param->updateGradient(attention_nodes[i].h_new_t.leftCols(current_minibatch_size),
													     minibatch_samples_no_negative.leftCols(current_minibatch_size),
													     d_Err_t_d_output.leftCols(current_minibatch_size));	
	 				}
					//Applying dropout
					output_dropout_layers[i].bProp(losses[i].d_Err_t_d_h_t);		
		   
	 			}
 	
		}
		
		/*********************************************************************************
		 * Backpropagation for decoder LSTM sequence
		 * Input: output_data -- target-side gold output
		 *        const_source_ctx -- source-side hidden states for attention backpropagation
		 *        sequence_token_num -- number of tokens for each sentence in minibatch
		 *        max_input_sent_len -- maximum sentence length in minibatch
		 *        gradient_check -- the flag indicating whether do graident check
		 *        norm_clipping -- which gradient control strategy is applied
		 * Output:
		 * 		  NULL
		*********************************************************************************/
	    template <typename DerivedOut> 
	    void bPropDecoder(const MatrixBase<DerivedOut> &output_data,
			 const vector< Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
			 const vector<int> &sequence_token_num,
			 unsigned int &max_input_sent_len,
			 bool gradient_check,
			 bool norm_clipping)
	    {	
			int current_minibatch_size = output_data.cols();
			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
			int output_sent_len = output_data.rows(); 
			
			//attention node and the first decoder LSTM node would propagate the errors to the source side
			//we initialize these variables before decoder backpropagation
			int enc_num_hidden = encoder_lstm_nodes[0].h_t.rows();
			for( int i = 0; i < max_input_sent_len; ++i ) {
				encoder_lstm_nodes[i].d_Err_t_to_n_d_h_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes[i].d_Err_t_to_n_d_c_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes_reverse[i].d_Err_t_to_n_d_h_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes_reverse[i].d_Err_t_to_n_d_c_t_from_dec.setZero(enc_num_hidden, minibatch_size);
			}

			//first getting decoder loss
			for (int i=output_sent_len-1; i>=0; i--) {
				if (i==0 && output_sent_len-1 > 0) {
					//if use feed input, the delta comes from two sources: output layer and next decoder LSTM node (feed input links the attention output to next decoder LSTM node) 
					Matrix<precision_type,Dynamic,Dynamic> d_Err_t_to_d_h_new_t;
					d_Err_t_to_d_h_new_t.setZero(losses[i].d_Err_t_d_h_t.rows(), minibatch_size);
					if( decoder_plstm->feed_input_flag )
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t + decoder_lstm_nodes[i+1].feed_input_node->d_Err_t_to_n_d_x_t;
					else
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t; 
					//attention node backpropagation
					attention_nodes[i].bProp(d_Err_t_to_d_h_new_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					//populate errors to source-side LSTM nodes
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);

					//decoder node backpropagation
					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropFeedInput(output_data.row(i),
								dummy_zero,
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);	
					}
					else {
						decoder_lstm_nodes[i].bProp(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);	
					}
					//populate the errors in the first decoder LSTM node to the source-side encoder LSTM nodes
					#pragma omp parallel for
					for( int j = 0; j < max_input_sent_len; ++j ){
						for( int k = 0; k < current_minibatch_size; ++k ){
							precision_type average_val = 1.0/sequence_token_num[k];
							encoder_lstm_nodes[j].d_Err_t_to_n_d_h_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne.block(0,k,enc_num_hidden,1);
							encoder_lstm_nodes[j].d_Err_t_to_n_d_c_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne.block(0,k,enc_num_hidden,1);
							encoder_lstm_nodes_reverse[j].d_Err_t_to_n_d_h_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne.block(enc_num_hidden,k,enc_num_hidden,1);
							encoder_lstm_nodes_reverse[j].d_Err_t_to_n_d_c_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne.block(enc_num_hidden,k,enc_num_hidden,1);
						}
					}
				} else if (i == output_sent_len-1) {	
					attention_nodes[i].bProp(losses[i].d_Err_t_d_h_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);
					
					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropFeedInput(output_data.row(i),
								attention_nodes[i-1].h_new_t,
								attention_nodes[i].d_Err_t_to_d_h_t,
								dummy_zero, //for the last lstm node, we just need to supply a bunch of zeros as the gradient of the future
								dummy_zero,
								gradient_check,
								norm_clipping);
					}
					else {
						decoder_lstm_nodes[i].bProp(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
								dummy_zero,
								gradient_check,
								norm_clipping);
					}
		
				} else if (i > 0) {
					Matrix<precision_type,Dynamic,Dynamic> d_Err_t_to_d_h_new_t;
					d_Err_t_to_d_h_new_t.setZero(losses[i].d_Err_t_d_h_t.rows(), minibatch_size);
					if( decoder_plstm->feed_input_flag )
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t + decoder_lstm_nodes[i+1].feed_input_node->d_Err_t_to_n_d_x_t;
					else
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t; 
					attention_nodes[i].bProp(d_Err_t_to_d_h_new_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);
					
					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropFeedInput(output_data.row(i),
								attention_nodes[i-1].h_new_t,
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);		
					}
					else {
						decoder_lstm_nodes[i].bProp(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);		
					}
				} 		   
			}
	  }
	  		
	    // Dense version (for standard log-likelihood)
		// Same as bPropDecoder except that bPropDecoderDropout deals with output layer dropout
	    template <typename DerivedOut> 
	    void bPropDecoderDropout(const MatrixBase<DerivedOut> &output_data,
			 const vector< Matrix<precision_type,Dynamic,Dynamic> > &const_source_ctx,
			 const vector<int> &sequence_token_num,
			 unsigned int &max_input_sent_len, 
			 bool gradient_check,
			 bool norm_clipping)
	    {			
			int current_minibatch_size = output_data.cols();
			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
			//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
			int output_sent_len = output_data.rows(); 

			int enc_num_hidden = encoder_lstm_nodes[0].h_t.rows();
			for( int i = 0; i < max_input_sent_len; ++i ) {
				encoder_lstm_nodes[i].d_Err_t_to_n_d_h_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes[i].d_Err_t_to_n_d_c_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes_reverse[i].d_Err_t_to_n_d_h_t_from_dec.setZero(enc_num_hidden, minibatch_size);
				encoder_lstm_nodes_reverse[i].d_Err_t_to_n_d_c_t_from_dec.setZero(enc_num_hidden, minibatch_size);
			}
			
			//first getting decoder loss
			for (int i=output_sent_len-1; i>=0; i--) {

				if (i==0 && output_sent_len-1 > 0) {
					
					Matrix<precision_type,Dynamic,Dynamic> d_Err_t_to_d_h_new_t;
					d_Err_t_to_d_h_new_t.setZero(losses[i].d_Err_t_d_h_t.rows(), minibatch_size);
					if( decoder_plstm->feed_input_flag )
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t + decoder_lstm_nodes[i+1].feed_input_node->d_Err_t_to_n_d_x_t;
					else
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t;
					attention_nodes[i].bProp(d_Err_t_to_d_h_new_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);

					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropDropoutFeedInput(output_data.row(i),
								dummy_zero,
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);	
					}
					else {
						decoder_lstm_nodes[i].bPropDropout(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);	
					}
					#pragma omp parallel for
					for( int j = 0; j < max_input_sent_len; ++j ){
						for( int k = 0; k < current_minibatch_size; ++k ){
							precision_type average_val = 1.0/sequence_token_num[k];
							encoder_lstm_nodes[j].d_Err_t_to_n_d_h_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne.block(0,k,enc_num_hidden,1);
							encoder_lstm_nodes[j].d_Err_t_to_n_d_c_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne.block(0,k,enc_num_hidden,1);
							encoder_lstm_nodes_reverse[j].d_Err_t_to_n_d_h_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne.block(enc_num_hidden,k,enc_num_hidden,1);
							encoder_lstm_nodes_reverse[j].d_Err_t_to_n_d_c_t_from_dec.col(k) += average_val * decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne.block(enc_num_hidden,k,enc_num_hidden,1);
						}
					}

				} else if (i == output_sent_len-1) {	

					attention_nodes[i].bProp(losses[i].d_Err_t_d_h_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);

					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropDropoutFeedInput(output_data.row(i),
								attention_nodes[i-1].h_new_t,
								attention_nodes[i].d_Err_t_to_d_h_t,
								dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
								dummy_zero,
								gradient_check,
								norm_clipping);
					}
					else {
						decoder_lstm_nodes[i].bPropDropout(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
								dummy_zero,
								gradient_check,
								norm_clipping);
					}
		
				} else if (i > 0) {
					
					Matrix<precision_type,Dynamic,Dynamic> d_Err_t_to_d_h_new_t;
					d_Err_t_to_d_h_new_t.setZero(losses[i].d_Err_t_d_h_t.rows(), minibatch_size);
					if( decoder_plstm->feed_input_flag )
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t + decoder_lstm_nodes[i+1].feed_input_node->d_Err_t_to_n_d_x_t;
					else
						d_Err_t_to_d_h_new_t = losses[i].d_Err_t_d_h_t;
					attention_nodes[i].bProp(d_Err_t_to_d_h_new_t, decoder_lstm_nodes[i].h_t, const_source_ctx, sequence_token_num, max_input_sent_len);
					attention_nodes[i].distributeErrToSource(encoder_lstm_nodes, encoder_lstm_nodes_reverse, sequence_token_num, max_input_sent_len);

					if( decoder_plstm->feed_input_flag ) {
						decoder_lstm_nodes[i].bPropDropoutFeedInput(output_data.row(i),
								attention_nodes[i-1].h_new_t,
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);		
					}
					else {
						decoder_lstm_nodes[i].bPropDropout(output_data.row(i),
								attention_nodes[i].d_Err_t_to_d_h_t,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);		
					}
				} 		   

			}
	  }

		/*********************************************************************************
		 * Backpropagation for encoder left-to-right LSTM sequence
		 * Input: input_data -- source-side input sentences
		 *        gradient_check -- the flag indicating whether do graident check
		 *        norm_clipping -- which gradient control strategy is applied
		 *        sequence_cont_indices -- the continuation vector for each sentence in minibatch
		 * Output:
		 * 		  NULL
		*********************************************************************************/
		template <typename DerivedIn, typename DerivedS>
			void bPropEncoder(const MatrixBase<DerivedIn> &input_data,
					bool gradient_check,
					bool norm_clipping,
					const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
			{	
				int current_minibatch_size = input_data.cols();
				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);

				int input_sent_len = input_data.rows();

				//Now backpropping through the left-to-right encoder

				//first getting decoder loss
				for (int i=input_sent_len-1; i>=0; i--) {
					// Now calling backprop for the LSTM nodes
					if (i == input_sent_len-1) {	
						encoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));			

						encoder_lstm_nodes[i].bProp(input_data.row(i),
								dummy_zero,
								dummy_zero,
								dummy_zero,
								gradient_check,
								norm_clipping);

					} else{
						encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));		

						encoder_lstm_nodes[i].bProp(input_data.row(i),
								dummy_zero,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);								   
					} 		   
				}
			} 	  


	  
		// Dense version (for standard log-likelihood)
		/*********************************************************************************
		 * Backpropagation for encoder right-to-left LSTM sequence
		 * Input: input_data -- source-side input sentences
		 *        gradient_check -- the flag indicating whether do graident check
		 *        norm_clipping -- which gradient control strategy is applied
		 *        sequence_cont_indices -- the continuation vector for each sentence in minibatch
		 * Output:
		 * 		  NULL
		*********************************************************************************/
		template <typename DerivedIn, typename DerivedS>
			void bPropEncoderReverse(const MatrixBase<DerivedIn> &input_data,
					bool gradient_check,
					bool norm_clipping,
					const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
			{	
				int current_minibatch_size = input_data.cols();
				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);

				int input_sent_len = input_data.rows();

				//Now backpropping through the encoder
				//first getting decoder loss
				for (int i=input_sent_len-1; i>=0; i--) {
					if (i == input_sent_len-1) {	
						encoder_lstm_nodes_reverse[i].filterStatesAndErrors(decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));			

						encoder_lstm_nodes_reverse[i].bProp(input_data.row(i),
								dummy_zero,
								dummy_zero,
								dummy_zero,
								gradient_check,
								norm_clipping);

					} else{
						encoder_lstm_nodes_reverse[i].filterStatesAndErrors(encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));		

						encoder_lstm_nodes_reverse[i].bProp(input_data.row(i),
								dummy_zero,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);								   
					} 		   
				}
			} 	  

		// Dense version (for standard log-likelihood)
		// Same as bPropEncoder except that bPropEncoderDropout applies dropout for input layer
		template <typename DerivedIn, typename DerivedS> 
			void bPropEncoderDropout(const MatrixBase<DerivedIn> &input_data,
					bool gradient_check,
					bool norm_clipping,
					const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
			{	
				int current_minibatch_size = input_data.cols();
				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);

				int input_sent_len = input_data.rows();

				//Now backpropping through the encoder
				for (int i=input_sent_len-1; i>=0; i--) {
					if (i == input_sent_len-1) {	
						encoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));			

						encoder_lstm_nodes[i].bPropDropout(input_data.row(i),
								dummy_zero,
								dummy_zero,
								dummy_zero,
								gradient_check,
								norm_clipping);

					} else{
						encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));		

						encoder_lstm_nodes[i].bPropDropout(input_data.row(i),
								dummy_zero,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);								   
					} 		   
				}
			} 


		// Dense version (for standard log-likelihood)
		// Same as bPropEncoderReverse except that bPropEncoderDropoutReverse applies dropout in input layer
		template <typename DerivedIn, typename DerivedS> 
			void bPropEncoderDropoutReverse(const MatrixBase<DerivedIn> &input_data,
					bool gradient_check,
					bool norm_clipping,
					const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
			{	
				int current_minibatch_size = input_data.cols();
				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);

				int input_sent_len = input_data.rows();

				//Now backpropping through the encoder
				for (int i=input_sent_len-1; i>=0; i--) {
					// Now calling backprop for the LSTM nodes
					if (i == input_sent_len-1) {	
						encoder_lstm_nodes_reverse[i].filterStatesAndErrors(decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
								decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));			

						encoder_lstm_nodes_reverse[i].bPropDropout(input_data.row(i),
								dummy_zero,
								dummy_zero,
								dummy_zero,
								gradient_check,
								norm_clipping);

					} else{
						encoder_lstm_nodes_reverse[i].filterStatesAndErrors(encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_h_tMinusOne,
								encoder_lstm_nodes_reverse[i+1].d_Err_t_to_n_d_c_tMinusOne,
								sequence_cont_indices.row(i));		

						encoder_lstm_nodes_reverse[i].bPropDropout(input_data.row(i),
								dummy_zero,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
								encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
								gradient_check,
								norm_clipping);								   
					} 		   
				}
			} 

		precision_type getGradSqdNorm(precision_type &grad_norm,
				loss_function_type loss_function,
				bool arg_run_lm){
			//First compute the norm of the gradients for norm scaling

			grad_norm += decoder_plstm->output_layer.getGradSqdNorm();
			if (arg_run_lm == 0) {
				grad_norm += encoder_plstm->getGradSqdNorm();
				grad_norm += encoder_plstm_reverse->getGradSqdNorm();
			}

			grad_norm += decoder_plstm->getGradSqdNorm();
			return(grad_norm);
		}

		/*********************************************************************************
		 * 
		 * update network params for left-to-right encoder, right-to-left encoder, decoder, attention and input output layers
		 * 		  
		*********************************************************************************/
		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				bool norm_clipping,
				precision_type norm_threshold,
				loss_function_type loss_function,
				bool arg_run_lm) {
			//first update the output layer params
			if (loss_function == LogLoss){
				decoder_plstm->output_layer.updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						norm_clipping,
						norm_threshold);			
			} else if (loss_function == NCELoss){
				decoder_plstm->output_layer.updateParamsNCE(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						norm_clipping,
						norm_threshold);	
			} else {
				cerr<<loss_function<<" is an invalid loss function type"<<endl;
				exit(0);
			}

			//update the encoder params
			if (arg_run_lm == 0) {
				encoder_plstm->updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						norm_clipping,
						norm_threshold);
				encoder_plstm_reverse->updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						norm_clipping,
						norm_threshold);
			}

			//update the decoder params
			decoder_plstm->updateParams(learning_rate,
					current_minibatch_size,
					momentum,
					L2_reg,
					norm_clipping,
					norm_threshold);												
		}

		void updateParams(precision_type learning_rate,
				int current_minibatch_size,
				precision_type momentum,
				precision_type L2_reg,
				precision_type grad_scale,
				loss_function_type loss_function,
				bool arg_run_lm) {
			if (loss_function == LogLoss){
				decoder_plstm->output_layer.updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						grad_scale);			
			} else if (loss_function == NCELoss){
				decoder_plstm->output_layer.updateParamsNCE(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						grad_scale);	
			} else {
				cerr<<loss_function<<" is an invalid loss function type"<<endl;
				exit(0);
			}

			if (arg_run_lm == 0) {
				encoder_plstm->updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						grad_scale);
				encoder_plstm_reverse->updateParams(learning_rate,
						current_minibatch_size,
						momentum,
						L2_reg,
						grad_scale);
			}

			decoder_plstm->updateParams(learning_rate,
					current_minibatch_size,
					momentum,
					L2_reg,
					grad_scale);												
		}
	  	  
		/*********************************************************************************
		 *
		 * compute probabilities according to the attention output
		 * Input: output -- target-side gold output
		 * 		  unigram -- unigram for NCE
		 * 		  num_noise_samples -- number of noise samples
		 * 		  rng -- random generator
		 * 		  loss-function -- loss function type such as tahn, sigmoid
		 * Output: log_likelihood -- loss
		 * 		  
		*********************************************************************************/
		template <typename DerivedOut, typename data_type>
			void computeProbs(const MatrixBase<DerivedOut> &output,
					multinomial<data_type> &unigram,
					int num_noise_samples,
					boost::random::mt19937 &rng,
					loss_function_type loss_function,
					precision_type &log_likelihood) 
			{	
				int current_minibatch_size = output.cols();

				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

				int sent_len = output.rows(); 
				for (int i=sent_len-1; i>=0; i--) {
					//First doing fProp for the output layer
					if (loss_function == LogLoss) {
						output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
								scores.leftCols(current_minibatch_size));

						//then compute the log loss of the objective
						precision_type minibatch_log_likelihood;
						start_timer(5);
						SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
								output.row(i), 
								probs, 
								minibatch_log_likelihood);
						stop_timer(5);
						log_likelihood += minibatch_log_likelihood;		
					} else if (loss_function == NCELoss) {
						//generate specific noise samples
						precision_type minibatch_log_likelihood;
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);

						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}

						//calculate the proabilities
						output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
								minibatch_samples_no_negative.leftCols(current_minibatch_size),
								scores);

						nce_loss.fProp(scores, 
								minibatch_samples,
								probs, 
								minibatch_log_likelihood,
								this->fixed_partition_function);
						log_likelihood += minibatch_log_likelihood;		
					}
				}
			}	  

		//similar to computeProbs differs in that computeProbsDropout uses output layer dropout
		template <typename DerivedOut>
			void computeProbsDropout(const MatrixBase<DerivedOut> &output,
					multinomial<data_size_t> &unigram,
					int num_noise_samples,
					boost::random::mt19937 &rng,
					loss_function_type loss_function,
					precision_type &log_likelihood) 
			{	
				int current_minibatch_size = output.cols();

				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

				int sent_len = output.rows(); 
				for (int i=sent_len-1; i>=0; i--) {
					//First doing fProp for the output layer
					if (loss_function == LogLoss) {
						output_dropout_layers[i].fProp(attention_nodes[i].h_new_t,rng);					
						output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), 
								scores.leftCols(current_minibatch_size));

						precision_type minibatch_log_likelihood;
						start_timer(5);
						SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
								output.row(i), 
								probs, 
								minibatch_log_likelihood);
						stop_timer(5);
						log_likelihood += minibatch_log_likelihood;		
					} else if (loss_function == NCELoss) {
						precision_type minibatch_log_likelihood;
						output_dropout_layers[i].fProp(decoder_lstm_nodes[i].h_t,rng);
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);

						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}
						output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
								minibatch_samples_no_negative.leftCols(current_minibatch_size),
								scores);

						nce_loss.fProp(scores, 
								minibatch_samples,
								probs, 
								minibatch_log_likelihood,
								this->fixed_partition_function);
						log_likelihood += minibatch_log_likelihood;							
					}
				}
			}	  
	  
		//compute log probability
		template <typename DerivedOut>
			void computeProbsLog(const MatrixBase<DerivedOut> &output,
					precision_type &log_likelihood) 
			{	
				int current_minibatch_size = output.cols();

				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

				int sent_len = output.rows(); 
				for (int i=sent_len-1; i>=0; i--) {
					//First doing fProp for the output layer
					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), scores);

					precision_type minibatch_log_likelihood;
					start_timer(5);
					SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
							output.row(i), 
							probs, 
							minibatch_log_likelihood);
					stop_timer(5);
					log_likelihood += minibatch_log_likelihood;		
				}

			}	  

		template <typename DerivedOut>
			void computeProbsLog(const MatrixBase<DerivedOut> &output,
					precision_type &log_likelihood,
					vector<precision_type> &sentence_probabilities) 
			{	
				int current_minibatch_size = output.cols();
				Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
				//Right now, we are setting the dimension of dummy zero to the output embedding dimension becase everything has the 
				//same dimension in and LSTM. this might not be a good idea
				dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

				int sent_len = output.rows(); 
				//first initializing the sentence log probabilities to 0
				sentence_probabilities = vector<precision_type> (current_minibatch_size,0.);
				for (int i=sent_len-1; i>=0; i--) {
					//First doing fProp for the output layer
					output_layer_node.param->fProp(attention_nodes[i].h_new_t.leftCols(current_minibatch_size), scores);

					//then compute the log loss of the objective
					precision_type minibatch_log_likelihood;
					start_timer(5);
					SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
							output.row(i), 
							probs, 
							minibatch_log_likelihood);
					//Now adding the sentence probabilities
					#pragma omp parallel for 
					for (int sent_index=0; sent_index<current_minibatch_size; sent_index++){
						int output_word_index = output(i,sent_index);
						sentence_probabilities[sent_index] += (output_word_index >= 0) ? probs(output_word_index,sent_index) : 0.;
					}
					stop_timer(5);
					log_likelihood += minibatch_log_likelihood;		
				}
			}		  

		//reset gradients for encoder and decoder models
		void resetGradient(){
			encoder_plstm->resetGradient();	
			encoder_plstm_reverse->resetGradient();	
			decoder_plstm->resetGradient();	
		}	
	
		//Use finite differences to do gradient check
		template <typename DerivedIn, typename DerivedOut, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
			void gradientCheck(const MatrixBase<DerivedIn> &input,
					const MatrixBase<DerivedIn> &input_reverse,
					const MatrixBase<DerivedOut> &decoder_input,
					const MatrixBase<DerivedOut> &decoder_output,
					const MatrixBase<DerivedC> &const_init_c,
					const MatrixBase<DerivedH> &const_init_h,
					multinomial<data_type> &unigram,
					int num_noise_samples,
					boost::random::mt19937 &rng,
					loss_function_type loss_function,
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices_reverse,
					const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
					const vector<int> &input_token_num,
					bool arg_run_lm,
					precision_type dropout_probability)
			{
				Matrix<precision_type,Dynamic,Dynamic> init_c = const_init_c;
				Matrix<precision_type,Dynamic,Dynamic> init_h = const_init_h;
				cerr<<"In gradient check"<<endl;

				//Check every dimension of all the parameters to make sure the gradient is fine
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->output_layer,
						"output_layer", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);	
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_h_to_c,
						"Decoder: W_h_to_c", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;		
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_h_to_f,
						"Decoder: W_h_to_f", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;										
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_h_to_o,
						"Decoder: W_h_to_o", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_h_to_i ,
						"Decoder: W_h_to_i", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_h_to_atten ,
						"Decoder: W_h_to_atten", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_ctx_to_atten ,
						"Decoder: W_ctx_to_atten", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				#ifdef PEEP						 
				init_c = const_init_c;
				init_h = const_init_h;		
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_c_to_o,
						"Decoder: W_c_to_o", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_c_to_f,
						"Decoder: W_c_to_f", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->W_c_to_i,
						"Decoder: W_c_to_i", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				#endif
				init_c = const_init_c;
				init_h = const_init_h;		
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->o_t,
						"Decoder: o_t",  
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->f_t,
						"Decoder: f_t",
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->i_t,
						"Decoder: i_t",
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->tanh_c_prime_t,
						"Decoder: tanh_c_prime_t", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);		
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						decoder_plstm->atten_t,
						"Decoder: atten_t", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);		
				//Doing gradient check for the input nodes
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_i,"Decoder: Standard_input_node: W_x_to_i", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);		
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_f,
						"Decoder: Standard_input_node: W_x_to_f", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);

				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_c,
						"Decoder: Standard_input_node: W_x_to_c", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;		 
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_o,
						"Decoder: Standard_input_node: W_x_to_o", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;		 
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<input_model_type*>(decoder_plstm->input))->input_layer,
						"Decoder: input_layer", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);							 
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<feed_input_model_type*>(decoder_plstm->feed_input))->W_x_to_i,"Decoder: Standard_input_node: W_x_to_i", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);		
				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<feed_input_model_type*>(decoder_plstm->feed_input))->W_x_to_f,
						"Decoder: Standard_input_node: W_x_to_f", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);

				init_c = const_init_c;
				init_h = const_init_h;
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<feed_input_model_type*>(decoder_plstm->feed_input))->W_x_to_c,
						"Decoder: Standard_input_node: W_x_to_c", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);
				init_c = const_init_c;
				init_h = const_init_h;		 
				paramGradientCheck(input, input_reverse,
						decoder_input, 
						decoder_output,
						(dynamic_cast<feed_input_model_type*>(decoder_plstm->feed_input))->W_x_to_o,
						"Decoder: Standard_input_node: W_x_to_o", 
						init_c,
						init_h,
						unigram,
						num_noise_samples,
						rng,
						loss_function,
						input_sequence_cont_indices,
						input_sequence_cont_indices_reverse,
						output_sequence_cont_indices,
						input_token_num,
						dropout_probability,
						arg_run_lm);


				if (arg_run_lm == 0) {	
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_h_to_c,
							"Encoder: W_h_to_c", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_h_to_f,
							"Encoder: W_h_to_f", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;										
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_h_to_o,
							"Encoder: W_h_to_o", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_h_to_i ,
							"Encoder: W_h_to_i", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					#ifdef PEEP
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_c_to_o,
							"Encoder: W_c_to_o", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->W_c_to_f,
							"Encoder: W_c_to_f", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse, 
							decoder_input, 
							decoder_output,
							encoder_plstm->W_c_to_i,
							"Encoder: W_c_to_i", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					#endif
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->o_t,
							"Encoder: o_t",  
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->f_t,
							"Encoder: f_t",
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm->i_t,
							"Encoder: i_t",
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse, 
							decoder_input, 
							decoder_output,
							encoder_plstm->tanh_c_prime_t,
							"Encoder: tanh_c_prime_t", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);		

					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_h_to_c,
							"Encoder: W_h_to_c", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_h_to_f,
							"Encoder: W_h_to_f", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;										
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_h_to_o,
							"Encoder: W_h_to_o", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_h_to_i ,
							"Encoder: W_h_to_i", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					#ifdef PEEP
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_c_to_o,
							"Encoder: W_c_to_o", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_c_to_f,
							"Encoder: W_c_to_f", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse, 
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->W_c_to_i,
							"Encoder: W_c_to_i", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					#endif
					init_c = const_init_c;
					init_h = const_init_h;		
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->o_t,
							"Encoder: o_t",  
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->f_t,
							"Encoder: f_t",
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->i_t,
							"Encoder: i_t",
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse, 
							decoder_input, 
							decoder_output,
							encoder_plstm_reverse->tanh_c_prime_t,
							"Encoder: tanh_c_prime_t", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);		
					//Doing gradient check for the input nodes
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_i,
							"Encoder: Standard_input_node: W_x_to_i", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);		
					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_f,
							"Encoder: Standard_input_node: W_x_to_f", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);

					init_c = const_init_c;
					init_h = const_init_h;
					paramGradientCheck(input, input_reverse,
							decoder_input, 
							decoder_output,
							(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_c,
							"Encoder: Standard_input_node: W_x_to_c", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);
					init_c = const_init_c;
					init_h = const_init_h;					 
					paramGradientCheck(input, input_reverse, 
							decoder_input, 
							decoder_output,
							(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_o,
							"Encoder: Standard_input_node: W_x_to_o", 
							init_c,
							init_h,
							unigram,
							num_noise_samples,
							rng,
							loss_function,
							input_sequence_cont_indices,
							input_sequence_cont_indices_reverse,
							output_sequence_cont_indices,
							input_token_num,
							dropout_probability,
							arg_run_lm);								 									 							 	
				}		
			}

		template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
			void paramGradientCheck(const MatrixBase<DerivedIn> &input,
					const MatrixBase<DerivedIn> &input_reverse,
					const MatrixBase<DerivedOut> &decoder_input,
					const MatrixBase<DerivedOut> &decoder_output,
					testParam &param,
					const string param_name,
					const MatrixBase<DerivedC> &init_c,
					const MatrixBase<DerivedH> &init_h, 
					multinomial<data_type> &unigram,
					int num_noise_samples,
					boost::random::mt19937 &rng,
					loss_function_type loss_function,			 
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices_reverse,
					const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
					const vector<int> &input_token_num,
					precision_type dropout_probability,
					bool arg_run_lm) {
				for(int row=0; row<param.rows(); row++){
					for (int col=0; col<param.cols(); col++){		
						getFiniteDiff(input, input_reverse,
								decoder_input,
								decoder_output, 
								param, 
								param_name, 
								row, 
								col, 
								init_c, 
								init_h,
								unigram,
								num_noise_samples,
								rng,
								loss_function,
								input_sequence_cont_indices,
								input_sequence_cont_indices_reverse,
								output_sequence_cont_indices,
								input_token_num,
								dropout_probability,
								arg_run_lm);
					}
				}
			}
	
		template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
			void getFiniteDiff(const MatrixBase<DerivedIn> &input,
					const MatrixBase<DerivedIn> &input_reverse,
					const MatrixBase<DerivedOut> &decoder_input,
					const MatrixBase<DerivedOut> &decoder_output,
					testParam &param,
					const string param_name,
					int row,
					int col,
					const MatrixBase<DerivedC> &const_init_c,
					const MatrixBase<DerivedH> &const_init_h,
					multinomial<data_type> &unigram,
					int num_noise_samples,
					boost::random::mt19937 &rng,
					loss_function_type loss_function,
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
					const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices_reverse,
					const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
					const vector<int> &input_token_num,
					precision_type dropout_probability,
					bool arg_run_lm) {
				Matrix<precision_type,Dynamic,Dynamic> init_c; 
				Matrix<precision_type,Dynamic,Dynamic> init_h;
				boost::random::mt19937 init_rng = rng;
				init_c = const_init_c;
				init_h = const_init_h;
				int rand_row = row;
				int rand_col = col;
				//First checking the gradient of the output word embeddings
				precision_type perturbation = 1e-3;
				param.changeRandomParam(perturbation, 
						rand_row,
						rand_col);
				//then do an fprop
				precision_type before_log_likelihood = 0;	

				int max_input_sent_len = input.rows();
				int num_hidden = const_init_h.rows();
				int current_minibatch_size = input.cols();
				vector<Matrix<precision_type,Dynamic,Dynamic> > source_ctx(105, Matrix<precision_type,Dynamic,Dynamic>());
				vector<Matrix<precision_type,Dynamic,Dynamic> > source_cell(105, Matrix<precision_type,Dynamic,Dynamic>());
				for( int i = 0; i < 105; ++i ){
					source_ctx[i].setZero(num_hidden*2, minibatch_size);
					source_cell[i].setZero(num_hidden*2, minibatch_size);
				}
				Matrix<precision_type,Dynamic,Dynamic> source_ctx_mean, source_cell_mean;

				if (dropout_probability > 0) {
					if (arg_run_lm == 0 ) {
						fPropEncoderDropout(input,
								init_c,
								init_h,
								input_sequence_cont_indices,
								init_rng);	
						init_c = const_init_c;
						init_h = const_init_h;
						fPropEncoderDropoutReverse(input_reverse,
								init_c,
								init_h,
								input_sequence_cont_indices_reverse,
								init_rng);	
					}

					//concatenate the forward and backward source contexts
					#pragma omp parallel for
					for( int i = 0; i < max_input_sent_len; ++i ){
						int j;
						for( j = 0; j < current_minibatch_size; ++j ){
							int padded_word_num = max_input_sent_len - input_token_num[j];
							if( i < padded_word_num ) {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[i].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
									encoder_lstm_nodes_reverse[i].c_t.col(j);
							}
							else {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[j].c_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].c_t.col(j);
							}
						}
						for( ; j < minibatch_size; ++j ){
							source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
								encoder_lstm_nodes_reverse[i].h_t.col(j);
							source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
								encoder_lstm_nodes_reverse[i].c_t.col(j);
						}
					}

					source_ctx_mean.setZero(num_hidden*2, minibatch_size);
					source_cell_mean.setZero(num_hidden*2, minibatch_size);
					for( int i = 0; i < max_input_sent_len; ++i ){
						source_ctx_mean += source_ctx[i];
						source_cell_mean += source_cell[i];
					}

					#pragma omp parallel for
					for( int i = 0; i < current_minibatch_size; ++i ){
						source_ctx_mean.col(i) = source_ctx_mean.col(i)/input_token_num[i];
						source_cell_mean.col(i) = source_cell_mean.col(i)/input_token_num[i];
					}

					fPropDecoderDropout(decoder_input,
							source_cell_mean,
							source_ctx_mean,
							source_ctx,
							output_sequence_cont_indices,
							input_token_num,
							max_input_sent_len,	
							init_rng);		
					computeProbsDropout(decoder_output,
							unigram,
							num_noise_samples,
							init_rng,
							loss_function,	
							before_log_likelihood);																					
				} else {	
					if (arg_run_lm == 0) {			
						fPropEncoder(input,
								init_c,
								init_h,
								input_sequence_cont_indices);
						init_c = const_init_c;
						init_h = const_init_h;
						fPropEncoderReverse(input_reverse,
								init_c,
								init_h,
								input_sequence_cont_indices_reverse);
					}

					//concatenate the forward and backward source contexts
					#pragma omp parallel for
					for( int i = 0; i < max_input_sent_len; ++i ){
						int j;
						for( j = 0; j < current_minibatch_size; ++j ){
							int padded_word_num = max_input_sent_len - input_token_num[j];
							if( i < padded_word_num ) {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[i].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
									encoder_lstm_nodes_reverse[i].c_t.col(j);
							}
							else {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[j].c_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].c_t.col(j);
							}
						}
						for( ; j < minibatch_size; ++j ){
							source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
								encoder_lstm_nodes_reverse[i].h_t.col(j);
							source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
								encoder_lstm_nodes_reverse[i].c_t.col(j);
						}
					}

					source_ctx_mean.setZero(num_hidden*2, minibatch_size);
					source_cell_mean.setZero(num_hidden*2, minibatch_size);
					for( int i = 0; i < max_input_sent_len; ++i ){
						source_ctx_mean += source_ctx[i];
						source_cell_mean += source_cell[i];
					}

					#pragma omp parallel for
					for( int i = 0; i < current_minibatch_size; ++i ){
						source_ctx_mean.col(i) = source_ctx_mean.col(i)/input_token_num[i];
						source_cell_mean.col(i) = source_cell_mean.col(i)/input_token_num[i];
					}


					fPropDecoder(decoder_input,
							source_cell_mean,
							source_ctx_mean,
							source_ctx,
							output_sequence_cont_indices,
							input_token_num,
							max_input_sent_len);	
					computeProbs(decoder_output,
							unigram,
							num_noise_samples,
							init_rng,
							loss_function,	
							before_log_likelihood);																
				}
				param.changeRandomParam(-2*perturbation, 
						rand_row,
						rand_col);		
				init_c = const_init_c;
				init_h = const_init_h;
				init_rng = rng;
				precision_type after_log_likelihood = 0;	
				if (dropout_probability > 0) {
					if (arg_run_lm == 0){
						fPropEncoderDropout(input,
								init_c,
								init_h,
								input_sequence_cont_indices,
								init_rng);	
						init_c = const_init_c;
						init_h = const_init_h;
						fPropEncoderDropoutReverse(input_reverse,
								init_c,
								init_h,
								input_sequence_cont_indices_reverse,
								init_rng);	
					}

					//concatenate the forward and backward source contexts
					#pragma omp parallel for
					for( int i = 0; i < max_input_sent_len; ++i ){
						int j;
						for( j = 0; j < current_minibatch_size; ++j ){
							int padded_word_num = max_input_sent_len - input_token_num[j];
							if( i < padded_word_num ) {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[i].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
									encoder_lstm_nodes_reverse[i].c_t.col(j);
							}
							else {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[j].c_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].c_t.col(j);
							}
						}
						for( ; j < minibatch_size; ++j ){
							source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
								encoder_lstm_nodes_reverse[i].h_t.col(j);
							source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
								encoder_lstm_nodes_reverse[i].c_t.col(j);
						}
					}

					source_ctx_mean.setZero(num_hidden*2, minibatch_size);
					source_cell_mean.setZero(num_hidden*2, minibatch_size);
					for( int i = 0; i < max_input_sent_len; ++i ){
						source_ctx_mean += source_ctx[i];
						source_cell_mean += source_cell[i];
					}

					#pragma omp parallel for
					for( int i = 0; i < current_minibatch_size; ++i ){
						source_ctx_mean.col(i) = source_ctx_mean.col(i)/input_token_num[i];
						source_cell_mean.col(i) = source_cell_mean.col(i)/input_token_num[i];
					}


					fPropDecoderDropout(decoder_input,
							source_cell_mean,
							source_ctx_mean,
							source_ctx,
							output_sequence_cont_indices,
							input_token_num,
							max_input_sent_len,
							init_rng);				
					computeProbsDropout(decoder_output,
							unigram,
							num_noise_samples,
							init_rng,
							loss_function,	
							after_log_likelihood);	

				} else {		
					if (arg_run_lm == 0){
						fPropEncoder(input,
								init_c,
								init_h,
								input_sequence_cont_indices);	
						init_c = const_init_c;
						init_h = const_init_h;
						fPropEncoderReverse(input_reverse,
								init_c,
								init_h,
								input_sequence_cont_indices_reverse);	
					}

					//concatenate the forward and backward source contexts
					#pragma omp parallel for
					for( int i = 0; i < max_input_sent_len; ++i ){
						int j;
						for( j = 0; j < current_minibatch_size; ++j ){
							int padded_word_num = max_input_sent_len - input_token_num[j];
							if( i < padded_word_num ) {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[i].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
									encoder_lstm_nodes_reverse[i].c_t.col(j);
							}
							else {
								source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].h_t.col(j);
								source_cell[i].col(j) << encoder_lstm_nodes[j].c_t.col(j),
									encoder_lstm_nodes_reverse[max_input_sent_len-i+padded_word_num-1].c_t.col(j);
							}
						}
						for( ; j < minibatch_size; ++j ){
							source_ctx[i].col(j) << encoder_lstm_nodes[i].h_t.col(j),
								encoder_lstm_nodes_reverse[i].h_t.col(j);
							source_cell[i].col(j) << encoder_lstm_nodes[i].c_t.col(j),
								encoder_lstm_nodes_reverse[i].c_t.col(j);
						}
					}

					source_ctx_mean.setZero(num_hidden*2, minibatch_size);
					source_cell_mean.setZero(num_hidden*2, minibatch_size);
					for( int i = 0; i < max_input_sent_len; ++i ){
						source_ctx_mean += source_ctx[i];
						source_cell_mean += source_cell[i];
					}

					#pragma omp parallel for
					for( int i = 0; i < current_minibatch_size; ++i ){
						source_ctx_mean.col(i) = source_ctx_mean.col(i)/input_token_num[i];
						source_cell_mean.col(i) = source_cell_mean.col(i)/input_token_num[i];
					}


					fPropDecoder(decoder_input,
							source_cell_mean,
							source_ctx_mean,
							source_ctx,
							output_sequence_cont_indices,	
							input_token_num,
							max_input_sent_len);	
					computeProbs(decoder_output,
							unigram,
							num_noise_samples,
							init_rng,
							loss_function,	
							after_log_likelihood);																		
				}

				//returning the parameter back to its own value
				param.changeRandomParam(perturbation , 
						rand_row,
						rand_col);			

				precision_type threshold = 1e-03;
				precision_type symmetric_finite_diff_grad = (before_log_likelihood-after_log_likelihood)/(2*perturbation);	
				precision_type graves_threshold = pow(10.0, (double) max(0.0, (double) ceil(log10(min(fabs(param.getGradient(rand_row,
												rand_col)), fabs(symmetric_finite_diff_grad)))))-6);
				precision_type gradient_diff =  symmetric_finite_diff_grad - param.getGradient(rand_row,
						rand_col);
				precision_type relative_error = fabs(param.getGradient(rand_row,rand_col)-symmetric_finite_diff_grad)/
					(fabs(param.getGradient(rand_row,rand_col)) + fabs(symmetric_finite_diff_grad));
				cerr<<std::setprecision(15);
				precision_type abs_diff = fabs(param.getGradient(rand_row,rand_col)-symmetric_finite_diff_grad);
				precision_type abs_max = max(fabs(param.getGradient(rand_row,rand_col)),fabs(symmetric_finite_diff_grad));
				if (gradient_diff > threshold || relative_error > threshold) {
					cerr<<"!!!GRADIENT CHECKING FAILED!!!"<<endl;
					cerr<<"Symmetric finite differences gradient is "<<	symmetric_finite_diff_grad<<endl;
					cerr<<"Algorithmic gradient is "<<param.getGradient(rand_row,rand_col)<<endl;					
					cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<endl;	

					cerr<<"The likelihoods before and after perturbation are "<< before_log_likelihood<<" "<<
						after_log_likelihood<<endl;
					cerr<<"Graves threshold is "<<graves_threshold<<endl;
					cerr<<"Relative error is "<<relative_error<<endl;
					exit(1);
				} else {
					cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<" and relative error is "<<relative_error<<endl;
					cerr<<"Symmetric finite differences gradient is "<<	symmetric_finite_diff_grad<<endl;
					cerr<<"Algorithmic gradient is "<<param.getGradient(rand_row,rand_col)<<endl;
					//cerr<<"Relative error is "<<relative error<<endl
				}

			}	
	};		
 
} // namespace nplm

#endif
