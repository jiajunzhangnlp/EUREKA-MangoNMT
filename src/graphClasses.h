//creating the structure of the nn in a graph that will help in performing backpropagation and forward propagation
#pragma once

#include <cstdlib>
#include "util.h"
#include "neuralClasses.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigen>


using namespace Eigen;
namespace nplm
{


using namespace Eigen;
	
	struct stateClipper{
		precision_type operator() (precision_type x) const { 
			return (precision_type) std::min(25., std::max(double(x),-25.));
		}
	};


	//Node class, the structure for each position for a specific layer
	template <class X>
	class Node {
		public:
			X * param; //what parameter is this
			Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix; //store forward propagation output
			Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix; //store backpropagation output
			int minibatch_size;

		public:
			Node() : param(NULL), minibatch_size(0) { }

			Node(X *input_param, int minibatch_size)
				: param(input_param),
				minibatch_size(minibatch_size)
			{
				resize(minibatch_size);
			}

			void resize(int minibatch_size)
			{
				this->minibatch_size = minibatch_size;
				if (param->n_outputs() != -1)
				{
					fProp_matrix.setZero(param->n_outputs(), minibatch_size);
				}
				if (param->n_inputs() != -1)
				{
					bProp_matrix.setZero(param->n_inputs(), minibatch_size);
				}
			}

			void resize() { resize(minibatch_size); }
	};

	//output loss node stores backpropagation results
	class Output_loss_node {
		int minibatch_size;
		public:
			Matrix<precision_type,Dynamic,Dynamic> d_Err_t_d_h_t;
			Output_loss_node() :minibatch_size(0),d_Err_t_d_h_t(Matrix<precision_type,Dynamic,Dynamic>()) {}

		void resize(int num_hidden,int minibatch_size) {
			this->minibatch_size = minibatch_size;
			d_Err_t_d_h_t.setZero(num_hidden, minibatch_size);
		}
	};

	//LSTM_node is the basic node for encoder and decoder LSTM sequence
	template <class input_node_type, class feed_input_node_type>
		class LSTM_node {
			int minibatch_size;
			public:
				//Each LSTM node has a bunch of nodes and temporary data structures
				Node<Linear_layer> W_h_to_i_node, W_h_to_f_node, W_h_to_c_node, W_h_to_o_node;//previous hidden state linked to different gates in current LSTM node
				Node<Linear_diagonal_layer> W_c_to_i_node, W_c_to_f_node, W_c_to_o_node;//previous cell state linked to different gates in current LSTM node
				Node<Hidden_layer> i_t_node,f_t_node,o_t_node,tanh_c_prime_t_node;//gate node
				Node<Activation_function> tanh_c_t_node;//c_t node
				Dropout_layer output_dropout_layer;//dropout layer for output


				//structures store forward propagation results and backpropagation results
				Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> h_t, c_t, c_t_minus_one, h_t_minus_one;
				Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_h_t,
					d_Err_t_to_n_d_h_t_from_dec,
					d_Err_t_to_n_d_c_t,
					d_Err_t_to_n_d_c_t_from_dec,
					d_Err_t_to_n_d_o_t,
					d_Err_t_to_n_d_f_t,
					d_Err_t_to_n_d_i_t,
					d_Err_t_to_n_d_tanh_c_t,
					d_Err_t_to_n_d_tanh_c_prime_t,
					d_Err_t_to_n_d_x_t,
					i_t_input_matrix,
					f_t_input_matrix,
					o_t_input_matrix,
					tanh_c_prime_t_input_matrix;

				Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic>	d_Err_t_to_n_d_h_tMinusOne,
					d_Err_t_to_n_d_c_tMinusOne;

				input_node_type *input_node; //input word --> word embedding --> forward and backpropagation resutls to different gates
				feed_input_node_type *feed_input_node; //attention output --> forward and backpropagation results to different gates

				bool feed_input_flag;
				void set_feed_input_flag(bool flag){ feed_input_flag = flag; }

				LSTM_node(): 
					W_h_to_i_node(),
					W_h_to_f_node(),
					W_h_to_c_node(),
					W_h_to_o_node(),
					W_c_to_i_node(),
					W_c_to_f_node(),
					W_c_to_o_node(),
					i_t_node(),
					f_t_node(),
					o_t_node(),
					tanh_c_prime_t_node(),
					tanh_c_t_node(),
					input_node(NULL),
					feed_input_flag(false),
					feed_input_node(NULL) {}

				//initialize the network params of the node with specific model params (all nodes share the same model params)
				LSTM_node(model &lstm, int minibatch_size): 
					W_h_to_i_node(&lstm.W_h_to_i, minibatch_size),
					W_h_to_f_node(&lstm.W_h_to_f, minibatch_size),
					W_h_to_c_node(&lstm.W_h_to_c, minibatch_size),
					W_h_to_o_node(&lstm.W_h_to_o, minibatch_size),
					W_c_to_i_node(&lstm.W_c_to_i, minibatch_size),
					W_c_to_f_node(&lstm.W_c_to_f, minibatch_size),
					W_c_to_o_node(&lstm.W_c_to_o, minibatch_size),
					i_t_node(&lstm.i_t,minibatch_size),
					f_t_node(&lstm.f_t,minibatch_size),
					o_t_node(&lstm.o_t,minibatch_size),
					tanh_c_prime_t_node(&lstm.tanh_c_prime_t,minibatch_size),
					tanh_c_t_node(&lstm.tanh_c_t,minibatch_size),
					input_node(NULL),
					feed_input_flag(false),
					feed_input_node(NULL)
				{
					this->minibatch_size = minibatch_size;
				}

				//Resizing all the parameters
				void resize(int minibatch_size){
					this->minibatch_size = minibatch_size;
					W_h_to_i_node.resize(minibatch_size);
					W_h_to_f_node.resize(minibatch_size);
					W_h_to_c_node.resize(minibatch_size);
					W_h_to_o_node.resize(minibatch_size);
					W_c_to_i_node.resize(minibatch_size);
					W_c_to_f_node.resize(minibatch_size);
					W_c_to_o_node.resize(minibatch_size);
					i_t_node.resize(minibatch_size);
					f_t_node.resize(minibatch_size);
					o_t_node.resize(minibatch_size);
					tanh_c_prime_t_node.resize(minibatch_size);

					//Resizing all the local node matrices
					h_t.setZero(W_h_to_i_node.param->n_inputs(),minibatch_size);
					c_t.setZero(W_c_to_i_node.param->n_inputs(),minibatch_size);
					h_t_minus_one.setZero(W_h_to_i_node.param->n_inputs(),minibatch_size);
					c_t_minus_one.setZero(W_c_to_i_node.param->n_inputs(),minibatch_size);
					d_Err_t_to_n_d_h_t.setZero(W_h_to_i_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_h_t_from_dec.setZero(W_h_to_i_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_c_t.setZero(W_c_to_i_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_c_t_from_dec.setZero(W_c_to_i_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_o_t.setZero(o_t_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_f_t.setZero(f_t_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_i_t.setZero(i_t_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_tanh_c_t.setZero(tanh_c_t_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_tanh_c_prime_t.setZero(tanh_c_prime_t_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_h_tMinusOne.setZero(W_h_to_i_node.param->n_outputs(),minibatch_size);
					d_Err_t_to_n_d_c_tMinusOne.setZero(W_c_to_i_node.param->n_outputs(),minibatch_size);
					i_t_input_matrix.setZero(i_t_node.param->n_inputs(),minibatch_size);
					f_t_input_matrix.setZero(f_t_node.param->n_inputs(),minibatch_size);
					o_t_input_matrix.setZero(o_t_node.param->n_inputs(),minibatch_size);
					tanh_c_prime_t_input_matrix.setZero(tanh_c_prime_t_node.param->n_inputs(),minibatch_size);
				} 

				void set_input_node(input_node_type &input_node){this->input_node = &input_node;}

				void set_feed_input_node(feed_input_node_type &feed_input_node){this->feed_input_node = &feed_input_node;}

				//input words --> word embeddings --> forward propagation results for different gates
				template<typename Derived> 
				void fPropInput(const MatrixBase<Derived> &data) {
					input_node->fProp(data);
				}

				template<typename Derived, typename Engine>
					void fPropInputDropout(const MatrixBase<Derived> &data,
							Engine &eng) {
						input_node->fPropDropout(data,eng);
					}	

				//attention output --> forward propagation results for different gates
				template<typename Derived> 
					void fPropFeedInput(const MatrixBase<Derived> &data) {
						feed_input_node->fProp(data);
					}

				//LSTM_node backpropagation: LSTM block backpropagation --> input node backpropagation
				template<typename DerivedData, typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
					void bProp(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
							const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
							const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
							bool gradient_check,
							bool norm_clipping){

						int current_minibatch_size = data.cols();

						bPropLSTMBlock(d_Err_t_d_h_t,
								d_Err_tPlusOne_to_n_d_c_t,
								d_Err_tPlusOne_to_n_d_h_t,
								gradient_check,
								norm_clipping,
								current_minibatch_size);	

						bPropInput(data);	
					}

				template<typename DerivedData, typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
					void bPropDropout(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
							const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
							const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
							bool gradient_check,
							bool norm_clipping){
						int current_minibatch_size = data.cols();

						bPropLSTMBlock(d_Err_t_d_h_t,
								d_Err_tPlusOne_to_n_d_c_t,
								d_Err_tPlusOne_to_n_d_h_t,
								gradient_check,
								norm_clipping,
								current_minibatch_size);	

						bPropInputDropout(data);			
					}

				//backpropagation for LSTM_node having feed input: LSTM block backpropagation --> feed input backpropagation --> input node backpropagation
				template<typename DerivedData, typename DerivedH, typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
					void bPropFeedInput(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedH> &h_new,
							const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
							const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
							const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
							bool gradient_check,
							bool norm_clipping){

						int current_minibatch_size = data.cols();

						bPropLSTMBlock(d_Err_t_d_h_t,
								d_Err_tPlusOne_to_n_d_c_t,
								d_Err_tPlusOne_to_n_d_h_t,
								gradient_check,
								norm_clipping,
								current_minibatch_size);	

						bPropFeedInput(h_new, current_minibatch_size);	
						bPropInput(data);	
					}

				template<typename DerivedData, typename DerivedH, typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
					void bPropDropoutFeedInput(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedH> &h_new,
							const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
							const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
							const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
							bool gradient_check,
							bool norm_clipping){
						int current_minibatch_size = data.cols();

						bPropLSTMBlock(d_Err_t_d_h_t,
								d_Err_tPlusOne_to_n_d_c_t,
								d_Err_tPlusOne_to_n_d_h_t,
								gradient_check,
								norm_clipping,
								current_minibatch_size);	

						bPropFeedInput(h_new, current_minibatch_size);			
						bPropInputDropout(data);			
					}


				//LSTM_node forward propagation: input data forward propagation --> LSTM block forward propagation
				template<typename Derived>
					void fProp(const MatrixBase<Derived> &data) {
						fPropInput(data);
						fPropLSTMBlock();
					}

				template<typename Derived, typename Engine>
					void fPropDropout(const MatrixBase<Derived> &data,
							Engine &eng) { 
						fPropInputDropout(data, eng);
						fPropLSTMBlock();
				}


				//LSTM_node (having feed input) forward propagation: input data forward propagation --> attention output forward propagation --> LSTM block forward
				template<typename Derived, typename DerivedH>
					void fPropFeedInput(const MatrixBase<Derived> &data, const MatrixBase<DerivedH> &h_new) { 
						fPropInput(data);
						fPropFeedInput(h_new);
						fPropLSTMBlock();
					}

				template<typename Derived, typename DerivedH, typename Engine>
					void fPropDropoutFeedInput(const MatrixBase<Derived> &data,
							const MatrixBase<DerivedH> &h_new,	
							Engine &eng) {
						fPropInputDropout(data, eng);
						fPropFeedInput(h_new);
						fPropLSTMBlock();
				}

				//input node backpropagation: update word embeddings
				template<typename DerivedData>	
					void bPropInput(const MatrixBase<DerivedData> &data){
						input_node->bProp(data,
								o_t_node.bProp_matrix,
								i_t_node.bProp_matrix,
								f_t_node.bProp_matrix,
								tanh_c_prime_t_node.bProp_matrix);		
					}

				template<typename DerivedData>	
					void bPropInputDropout(const MatrixBase<DerivedData> &data){
						input_node->bPropDropout(data,
								o_t_node.bProp_matrix,
								i_t_node.bProp_matrix,
								f_t_node.bProp_matrix,
								tanh_c_prime_t_node.bProp_matrix);		
					}

				//feed input node backpropagation: update feed_input_node params and calculate the gradient to attention output
				template<typename DerivedData>	
					void bPropFeedInput(const MatrixBase<DerivedData> &data, const int &current_minibatch_size){
						feed_input_node->bProp(data,
								o_t_node.bProp_matrix,
								i_t_node.bProp_matrix,
								f_t_node.bProp_matrix,
								tanh_c_prime_t_node.bProp_matrix,
								current_minibatch_size);		
					}

			//if PEEP is defined, the LSTM block will activate previous cell to link input gate, output gate and forget gate
			#ifdef PEEP
			void fPropLSTMBlock() {	
				//How much to remember
				W_h_to_i_node.param->fProp(h_t_minus_one,W_h_to_i_node.fProp_matrix);
				W_c_to_i_node.param->fProp(c_t_minus_one,W_c_to_i_node.fProp_matrix);
				i_t_input_matrix.noalias() = input_node->W_x_to_i_node.fProp_matrix + W_h_to_i_node.fProp_matrix + W_c_to_i_node.fProp_matrix;
				if( feed_input_flag )
					i_t_input_matrix.noalias() += feed_input_node->W_x_to_i_node.fProp_matrix;

				i_t_node.param->fProp(i_t_input_matrix,
						i_t_node.fProp_matrix);

				//How much to forget
				W_h_to_f_node.param->fProp(h_t_minus_one,W_h_to_f_node.fProp_matrix);
				W_c_to_f_node.param->fProp(c_t_minus_one,W_c_to_f_node.fProp_matrix);
				f_t_input_matrix.noalias() = input_node->W_x_to_f_node.fProp_matrix + W_h_to_f_node.fProp_matrix + W_c_to_f_node.fProp_matrix;
				if( feed_input_flag )
					f_t_input_matrix.noalias() += feed_input_node->W_x_to_f_node.fProp_matrix;
				f_t_node.param->fProp(f_t_input_matrix,
						f_t_node.fProp_matrix);

				//Generating the current cell value
				W_h_to_c_node.param->fProp(h_t_minus_one,W_h_to_c_node.fProp_matrix);	
				tanh_c_prime_t_input_matrix.noalias() = input_node->W_x_to_c_node.fProp_matrix + W_h_to_c_node.fProp_matrix;
				if( feed_input_flag )
					tanh_c_prime_t_input_matrix.noalias() += feed_input_node->W_x_to_c_node.fProp_matrix;
				tanh_c_prime_t_node.param->fProp(tanh_c_prime_t_input_matrix,
						tanh_c_prime_t_node.fProp_matrix);

				c_t.array() = f_t_node.fProp_matrix.array()*c_t_minus_one.array() + 
					i_t_node.fProp_matrix.array()*tanh_c_prime_t_node.fProp_matrix.array();

				//How much to scale the output
				W_h_to_o_node.param->fProp(h_t_minus_one,W_h_to_o_node.fProp_matrix);
				W_c_to_o_node.param->fProp(c_t,W_c_to_o_node.fProp_matrix);
				o_t_input_matrix.noalias() = input_node->W_x_to_o_node.fProp_matrix +  
					W_h_to_o_node.fProp_matrix + 
					W_c_to_o_node.fProp_matrix;

				if( feed_input_flag )
					o_t_input_matrix.noalias() += feed_input_node->W_x_to_o_node.fProp_matrix;

				o_t_node.param->fProp(o_t_input_matrix,
						o_t_node.fProp_matrix);	

				tanh_c_t_node.param->fProp(c_t,tanh_c_t_node.fProp_matrix);

				h_t.array() = o_t_node.fProp_matrix.array()*tanh_c_t_node.fProp_matrix.array();		
			}


			template<typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
				void bPropLSTMBlock (const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
						const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
						const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
						bool gradient_check,
						bool norm_clipping,
						int current_minibatch_size) {
					Matrix<precision_type,Dynamic,Dynamic> dummy_matrix;

					//NOTE: d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt h_t. 
					//Similarly, d_Err_t_to_n_d_c_t is read as derivative of Error function from time t to n wrt c_t. 
					//This is a slight abuse of notation. In our case, since we're maximizing log likelihood, we're taking derivatives of the negative of the 
					//error function, which is the cross entropy.

					//Summing up the error derivatives for h_t from t+1,..,n and from t
					//if it is encoder LSTM node, there will be derivatives from decoder
					d_Err_t_to_n_d_h_t = d_Err_t_to_n_d_h_t_from_dec + d_Err_t_d_h_t + d_Err_tPlusOne_to_n_d_h_t;

					//Derivative wrt o_t
					d_Err_t_to_n_d_o_t.array() = d_Err_t_to_n_d_h_t.array()*tanh_c_t_node.fProp_matrix.array();

					o_t_node.param->bProp(d_Err_t_to_n_d_o_t,
							o_t_node.bProp_matrix,
							dummy_matrix,
							o_t_node.fProp_matrix);// the third	field does not matter. Its a dummy matrix

					//Summing up the error derivatives for h_t from t+1,..,n and from t					  
					d_Err_t_to_n_d_tanh_c_t.array() = d_Err_t_to_n_d_h_t.array() * o_t_node.fProp_matrix.array();

					tanh_c_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_t,
							tanh_c_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_t_node.fProp_matrix);

					W_c_to_o_node.param->bProp(o_t_node.bProp_matrix,
							W_c_to_o_node.bProp_matrix);

					d_Err_t_to_n_d_c_t =  d_Err_t_to_n_d_c_t_from_dec + tanh_c_t_node.bProp_matrix + W_c_to_o_node.bProp_matrix + d_Err_tPlusOne_to_n_d_c_t;

					//Error derivatives for f_t
					d_Err_t_to_n_d_f_t.array() = d_Err_t_to_n_d_c_t.array()*c_t_minus_one.array();
					f_t_node.param->bProp(d_Err_t_to_n_d_f_t,
							f_t_node.bProp_matrix,
							dummy_matrix,
							f_t_node.fProp_matrix);

					//Error derivatives for i_t
					d_Err_t_to_n_d_i_t.array() = d_Err_t_to_n_d_c_t.array()*tanh_c_prime_t_node.fProp_matrix.array();
					i_t_node.param->bProp(d_Err_t_to_n_d_i_t,
							i_t_node.bProp_matrix,
							dummy_matrix,
							i_t_node.fProp_matrix);	

					//Error derivatives for c_prime_t
					d_Err_t_to_n_d_tanh_c_prime_t.array() = d_Err_t_to_n_d_c_t.array()*i_t_node.fProp_matrix.array();
					tanh_c_prime_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_prime_t,
							tanh_c_prime_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_prime_t_node.fProp_matrix);	

					//Error derivatives for h_t_minus_one
					W_h_to_o_node.param->bProp(o_t_node.bProp_matrix,
							W_h_to_o_node.bProp_matrix);
					W_h_to_f_node.param->bProp(f_t_node.bProp_matrix,
							W_h_to_f_node.bProp_matrix);
					W_h_to_i_node.param->bProp(i_t_node.bProp_matrix,
							W_h_to_i_node.bProp_matrix);

					W_h_to_c_node.param->bProp(tanh_c_prime_t_node.bProp_matrix,
							W_h_to_c_node.bProp_matrix);
					d_Err_t_to_n_d_h_tMinusOne = W_h_to_o_node.bProp_matrix + 
						W_h_to_f_node.bProp_matrix +
						W_h_to_i_node.bProp_matrix +
						W_h_to_c_node.bProp_matrix;		

					//Error derivatives for c_t_minus_one
					W_c_to_f_node.param->bProp(f_t_node.bProp_matrix,
							W_c_to_f_node.bProp_matrix);
					W_c_to_i_node.param->bProp(i_t_node.bProp_matrix,
							W_c_to_i_node.bProp_matrix);	

					d_Err_t_to_n_d_c_tMinusOne = (d_Err_t_to_n_d_c_t.array()*f_t_node.fProp_matrix.array()).matrix()+
						W_c_to_f_node.bProp_matrix +
						W_c_to_i_node.bProp_matrix;

					//Computing gradients of the paramters
					//Derivative of weights out of h_t
					W_h_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));
					W_h_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));
					W_h_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));		
					W_h_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));

					//Derivative of weights out of c_t and c_t_minus_one
					W_c_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
							this->c_t.leftCols(current_minibatch_size));
					W_c_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
							c_t_minus_one.leftCols(current_minibatch_size));
					W_c_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
							c_t_minus_one.leftCols(current_minibatch_size));		

					// Updating the gradient of the hidden layer biases									
					o_t_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size));
					f_t_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size));
					i_t_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size));
					tanh_c_prime_t_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size));	
				}

			#else
			//fProp without peeps
			void fPropLSTMBlock() {
				W_h_to_i_node.param->fProp(h_t_minus_one,W_h_to_i_node.fProp_matrix);
				W_h_to_c_node.param->fProp(h_t_minus_one,W_h_to_c_node.fProp_matrix);	
				W_h_to_o_node.param->fProp(h_t_minus_one,W_h_to_o_node.fProp_matrix);
				W_h_to_f_node.param->fProp(h_t_minus_one,W_h_to_f_node.fProp_matrix);


				i_t_input_matrix.noalias() = input_node->W_x_to_i_node.fProp_matrix + W_h_to_i_node.fProp_matrix;
				if( feed_input_flag )
					i_t_input_matrix.noalias() += feed_input_node->W_x_to_i_node.fProp_matrix;

				f_t_input_matrix.noalias() = input_node->W_x_to_f_node.fProp_matrix + W_h_to_f_node.fProp_matrix;
				if( feed_input_flag )
					f_t_input_matrix.noalias() += feed_input_node->W_x_to_f_node.fProp_matrix;

				//Computing input and forget gates
				i_t_node.param->fProp(i_t_input_matrix,
						i_t_node.fProp_matrix);

				//How much to forget					
				f_t_node.param->fProp(f_t_input_matrix,
						f_t_node.fProp_matrix);		

				//computing c_prime_t
				tanh_c_prime_t_input_matrix.noalias() = input_node->W_x_to_c_node.fProp_matrix + W_h_to_c_node.fProp_matrix;
				if( feed_input_flag ) {
					tanh_c_prime_t_input_matrix.noalias() += feed_input_node->W_x_to_c_node.fProp_matrix;
				}
				tanh_c_prime_t_node.param->fProp(tanh_c_prime_t_input_matrix,
						tanh_c_prime_t_node.fProp_matrix);

				//Computing the current cell value
				c_t.array() = f_t_node.fProp_matrix.array()*c_t_minus_one.array() + 
					i_t_node.fProp_matrix.array()*tanh_c_prime_t_node.fProp_matrix.array();

				//How much to scale the output
				o_t_input_matrix.noalias() = input_node->W_x_to_o_node.fProp_matrix +  
					W_h_to_o_node.fProp_matrix;
				if( feed_input_flag )
					o_t_input_matrix.noalias() += feed_input_node->W_x_to_o_node.fProp_matrix;

				//Computing the output gate
				//std::cerr<<"o t input matrix is "<<o_t_input_matrix<<std::endl;
				o_t_node.param->fProp(o_t_input_matrix,
						o_t_node.fProp_matrix);	

				//computing the hidden layer
				tanh_c_t_node.param->fProp(c_t,tanh_c_t_node.fProp_matrix);
				//<<"tanh_c_t_node.fProp_matrix is "<<tanh_c_t_node.fProp_matrix<<endl;
				h_t.array() = o_t_node.fProp_matrix.array()*tanh_c_t_node.fProp_matrix.array();		

			}

			template<typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
				void bPropLSTMBlock (const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
						const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
						const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
						bool gradient_check,
						bool norm_clipping,
						int current_minibatch_size) {	   
					Matrix<precision_type,Dynamic,Dynamic> dummy_matrix;

					//NOTE: d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt h_t. 
					//Similarly, d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt c_t. 
					//This is a slight abuse of notation. In our case, since we're maximizing log likelihood, we're taking derivatives of the negative of the 
					//error function, which is the cross entropy.

					//Error derivatives for h_t
					d_Err_t_to_n_d_h_t = d_Err_t_to_n_d_h_t_from_dec + d_Err_t_d_h_t + d_Err_tPlusOne_to_n_d_h_t;

					//Error derivativs for o_t
					d_Err_t_to_n_d_o_t.array() = d_Err_t_to_n_d_h_t.array()*tanh_c_t_node.fProp_matrix.array();

					o_t_node.param->bProp(d_Err_t_to_n_d_o_t,
							o_t_node.bProp_matrix,
							dummy_matrix,
							o_t_node.fProp_matrix);// the third	 field does not matter. Its a dummy matrix

					//Error derivatives for tanh_c_t					   
					d_Err_t_to_n_d_tanh_c_t.array() = d_Err_t_to_n_d_h_t.array() * o_t_node.fProp_matrix.array();
					tanh_c_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_t,
							tanh_c_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_t_node.fProp_matrix);

					//Error derivatives for c_t
					d_Err_t_to_n_d_c_t.noalias() =  d_Err_t_to_n_d_c_t_from_dec + tanh_c_t_node.bProp_matrix + d_Err_tPlusOne_to_n_d_c_t;

					//Error derivatives for f_t
					d_Err_t_to_n_d_f_t.array() = d_Err_t_to_n_d_c_t.array()*c_t_minus_one.array();
					f_t_node.param->bProp(d_Err_t_to_n_d_f_t,
							f_t_node.bProp_matrix,
							dummy_matrix,
							f_t_node.fProp_matrix);

					//Error derivatives for i_t
					d_Err_t_to_n_d_i_t.array() = d_Err_t_to_n_d_c_t.array()*tanh_c_prime_t_node.fProp_matrix.array();
					i_t_node.param->bProp(d_Err_t_to_n_d_i_t,
							i_t_node.bProp_matrix,
							dummy_matrix,
							i_t_node.fProp_matrix);	

					//Error derivatives for c_prime_t
					d_Err_t_to_n_d_tanh_c_prime_t.array() = d_Err_t_to_n_d_c_t.array()*i_t_node.fProp_matrix.array();

					tanh_c_prime_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_prime_t,
							tanh_c_prime_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_prime_t_node.fProp_matrix);	

					//Error derivatives for h_t_minus_one
					W_h_to_o_node.param->bProp(o_t_node.bProp_matrix,
							W_h_to_o_node.bProp_matrix);
					W_h_to_f_node.param->bProp(f_t_node.bProp_matrix,
							W_h_to_f_node.bProp_matrix);
					W_h_to_i_node.param->bProp(i_t_node.bProp_matrix,
							W_h_to_i_node.bProp_matrix);
					W_h_to_c_node.param->bProp(tanh_c_prime_t_node.bProp_matrix,
							W_h_to_c_node.bProp_matrix);
					d_Err_t_to_n_d_h_tMinusOne = W_h_to_o_node.bProp_matrix + 
						W_h_to_f_node.bProp_matrix +
						W_h_to_i_node.bProp_matrix +
						W_h_to_c_node.bProp_matrix;		

					d_Err_t_to_n_d_c_tMinusOne.noalias() = (d_Err_t_to_n_d_c_t.array()*f_t_node.fProp_matrix.array()).matrix();

					//Computing gradients of the paramters
					//Derivative of weights out of h_t
					W_h_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));
					W_h_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));
					W_h_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));		
					W_h_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size),
							h_t_minus_one.leftCols(current_minibatch_size));

					// Updating the gradient of the hidden layer biases									
					o_t_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size));
					f_t_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size));
					i_t_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size));
					tanh_c_prime_t_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size));		   	
				}	

				#endif


				//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block
				template <typename DerivedH, typename DerivedC>//, typename DerivedS>
					void copyToHiddenStates(const MatrixBase<DerivedH> &h_t_minus_one,
							const MatrixBase<DerivedC> &c_t_minus_one) {
						int current_minibatch_size = h_t_minus_one.cols();	
						#pragma omp parallel for 
						for (int index=0; index<current_minibatch_size; index++){ 
							this->h_t_minus_one.col(index) = h_t_minus_one.col(index);
							this->c_t_minus_one.col(index) = c_t_minus_one.col(index);
						}
					}

				//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block
				template <typename DerivedH, typename DerivedC , typename DerivedS>
					static void filterStatesAndErrors(const MatrixBase<DerivedH> &from_h_matrix,
							const MatrixBase<DerivedC> &from_c_matrix,
							const MatrixBase<DerivedH> &const_to_h_matrix,
							const MatrixBase<DerivedC> &const_to_c_matrix,
							const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) {
						int current_minibatch_size = sequence_cont_indices.cols();	
						UNCONST(DerivedC, const_to_c_matrix, to_c_matrix);
						UNCONST(DerivedH, const_to_h_matrix, to_h_matrix);

						#pragma omp parallel for 
						for (int index=0; index<current_minibatch_size; index++){ 
							if (sequence_cont_indices(index) == 0) {
								to_h_matrix.col(index).setZero(); 			
								to_c_matrix.col(index).setZero();
							} else {
								to_h_matrix.col(index) = from_h_matrix.col(index);
								to_c_matrix.col(index) = from_c_matrix.col(index);
							}
						}	
					}

					//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block/
					//It also takes a list that says where the new hidden states must be obtained from. This is very useful for beam search.
					//In beam search, the hidden states must come from the current k-best sequence. The current k-best sequence might be continuations
					//of the same item or different items in the previous k-best sequence and thus we can't just copy over the previous hidden states
					//because some of the k-best sequences corresponding to previous hidden states might have died. 
					//TODO: should the indices be a vector ? 
					template <typename DerivedH, typename DerivedC , typename indexType>
						static void copyKBestHiddenStates(const MatrixBase<DerivedH> &from_h_matrix,
								const MatrixBase<DerivedC> &from_c_matrix,
								const MatrixBase<DerivedH> &const_to_h_matrix,
								const MatrixBase<DerivedC> &const_to_c_matrix,
								const vector<indexType> &from_indices) {
							int current_minibatch_size = from_indices.size();

							UNCONST(DerivedC, const_to_c_matrix, to_c_matrix);
							UNCONST(DerivedH, const_to_h_matrix, to_h_matrix);

							#pragma omp parallel for 
							for (int index=0; index<current_minibatch_size; index++){ 
								to_h_matrix.col(index) = from_h_matrix.col(from_indices.at(index)); 			
								to_c_matrix.col(index) = from_c_matrix.col(from_indices.at(index));
							}	
						}

					//For stability, the gradient of the inputs of the loss to the LSTM is clipped, that is before applying the tanh and sigmoid
					//nonlinearities 
					void clipGradient(){}

					void resetGradient(){
					}	

					//Question, should the state gather be a part of 
					template<typename DerivedO, 
						typename DerivedF, 
						typename DerivedI, 
						typename DerivedH, 
						typename DerivedC>
							void getInternals(const MatrixBase<DerivedH> &const_get_h_t,
									const MatrixBase<DerivedC>   &const_get_c_t,
									const MatrixBase<DerivedF>   &const_get_f_t,
									const MatrixBase<DerivedI>   &const_get_i_t,
									const MatrixBase<DerivedO>   &const_get_o_t,
									const int sent_index){
								UNCONST(DerivedH, const_get_h_t, get_h_t);
								UNCONST(DerivedH, const_get_c_t, get_c_t);
								UNCONST(DerivedH, const_get_f_t, get_f_t);
								UNCONST(DerivedH, const_get_i_t, get_i_t);
								UNCONST(DerivedH, const_get_o_t, get_o_t);
								get_h_t = h_t.col(sent_index);
								get_c_t = c_t.col(sent_index);
								get_f_t = f_t_node.fProp_matrix.col(sent_index);
								get_i_t = i_t_node.fProp_matrix.col(sent_index);	
								get_o_t = o_t_node.fProp_matrix.col(sent_index);
							}


				};


			//Attention_node: takes decoder LSTM hidden state and source-side encoder LSTM hidden states as input
			//                forward --> calculate attention resutls
			//                backward --> calculate derivatives for inputs
			template <class input_node_type, class feed_input_node_type>
				class Attention_node {
					int minibatch_size;
					public:
					//sim_scores[i] = encoder_lstm_nodes[i].h_t \dot h_t (we use dot product here)
					//ctx_alpha[i] = softmax(sim_scores[i])
					//ctx_t = \sum ctx_alpha[i] \dot encoder_lstm_nodes[i].h_t
					//r_h_t = W_h_to_atten_node.param * h_t
					//r_ctx_t = W_ctx_to_atten_node.param * ctx_t
					//atten_input_matrix = r_h_t + r_ctx_t
					//atten_t = tahn(atten_inpu_matrix+bias) performed by atten_t_node
					Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> ctx_t, h_new_t; 
					Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> sim_scores, ctx_alpha;
					Node<Linear_layer> W_h_to_atten_node, W_ctx_to_atten_node;
					Node<Hidden_layer> atten_t_node;

					vector<Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> > d_Err_t_to_d_h_s_all;

					Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_d_alpha, d_Err_t_to_d_sim;

					Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_d_h_t,
						atten_t_input_matrix;
					Attention_node() {}

					//initialize the attention node network params with the specific model params
					Attention_node(model &lstm, int minibatch_size):
						W_h_to_atten_node(&lstm.W_h_to_atten, minibatch_size),
						W_ctx_to_atten_node(&lstm.W_ctx_to_atten, minibatch_size),
						atten_t_node(&lstm.atten_t, minibatch_size)
					{
						resize(minibatch_size);
					}

					//Resizing all the parameters
					void resize(int minibatch_size){
						this->minibatch_size = minibatch_size;
						W_h_to_atten_node.resize(minibatch_size);
						W_ctx_to_atten_node.resize(minibatch_size);
						atten_t_node.resize(minibatch_size);

						//Resizing all the local node matrices
						ctx_t.setZero(W_ctx_to_atten_node.param->n_inputs(),minibatch_size);
						h_new_t.setZero(atten_t_node.param->n_outputs(),minibatch_size);
						d_Err_t_to_d_h_t.setZero(W_h_to_atten_node.param->n_inputs(),minibatch_size);
						atten_t_input_matrix.setZero(atten_t_node.param->n_inputs(),minibatch_size);
					} 

					//after attention node backpropagation resulting derivatives to source-side hidden states, distributing them to left-to-right and right-to-left encoder LSTM sequence
					void distributeErrToSource(vector<LSTM_node<input_node_type,feed_input_node_type> > &encoder_lstm_nodes, vector<LSTM_node<input_node_type,feed_input_node_type> > &encoder_lstm_nodes_reverse, const vector<int> &sequence_token_num, const unsigned int &max_input_sent_len) {
						int hidden_dim = encoder_lstm_nodes[0].d_Err_t_to_n_d_h_t_from_dec.rows();
						unsigned int sen_len = max_input_sent_len;
						int current_minibatch_size = sequence_token_num.size();

						//#pragma omp parallel for
						for( int i = 0; i < sen_len; ++i ){
							for( int j = 0; j < current_minibatch_size; ++j ){
								int padded_word_num = sen_len - sequence_token_num[j];
								if( i < padded_word_num ) {
									encoder_lstm_nodes[i].d_Err_t_to_n_d_h_t_from_dec.col(j) += d_Err_t_to_d_h_s_all[i].block(0,0,hidden_dim,current_minibatch_size).col(j);
									encoder_lstm_nodes_reverse[i].d_Err_t_to_n_d_h_t_from_dec.col(j) += d_Err_t_to_d_h_s_all[i].block(hidden_dim,0,hidden_dim,current_minibatch_size).col(j);
								}
								else {
									encoder_lstm_nodes[i].d_Err_t_to_n_d_h_t_from_dec.col(j) += d_Err_t_to_d_h_s_all[i].block(0,0,hidden_dim,current_minibatch_size).col(j);
									encoder_lstm_nodes_reverse[sen_len-i+padded_word_num-1].d_Err_t_to_n_d_h_t_from_dec.col(j) += d_Err_t_to_d_h_s_all[i].block(hidden_dim,0,hidden_dim,current_minibatch_size).col(j);
								}
							}
						}
					}


					//attention node forward: decoder LSTM node hidden state h_t and source-side hidden states h_s_all are inputs, output the attention hidden state
					template<typename Derived>
						void fProp(const MatrixBase<Derived> &h_t, const vector<Eigen::Matrix<precision_type,Eigen::Dynamic, Eigen::Dynamic> > &h_s_all, const vector<int> &sequence_token_num, const unsigned int &max_input_sent_len) { 
							unsigned int h_s_size = max_input_sent_len;
							sim_scores.setZero(h_s_size,minibatch_size);
							ctx_alpha.setZero(h_s_size,minibatch_size);

							int current_minibatch_size = sequence_token_num.size();

							//sim_scores[i] = encoder_lstm_nodes[i].h_t \dot h_t
							#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){ //h_s_all[i] dim: hidden_size*minibatch_size
								//h_s_all[i] dim: hidden_size*minibatch_size
								//h_t dim: hidden_size*minibatch_size
								for( int j = 0; j < current_minibatch_size; ++j ){
									sim_scores(i, j) = h_t.col(j).dot(h_s_all[i].col(j));
								}
							}

							//ctx_alpha[i] = softmax(sim_scores[i])
							#pragma omp parallel for
							for( int i = 0; i < current_minibatch_size; ++i ){
								int padded_word_num = h_s_size - sequence_token_num[i];
								int max_id;
								precision_type max_value = sim_scores.col(i).maxCoeff(&max_id);
								ctx_alpha.col(i).array() = sim_scores.col(i).array() - max_value;
								ctx_alpha.col(i) = ctx_alpha.col(i).array().exp();
								for( int j = 0; j < padded_word_num; ++j ){
									ctx_alpha(j,i) = 0.;
								}
								precision_type col_sum = ctx_alpha.col(i).sum();
								ctx_alpha.col(i) /= col_sum;
							}
							
							ctx_t.setZero(W_ctx_to_atten_node.param->n_inputs(),minibatch_size);
							//ctx_t = \sum ctx_alpha[i] \dot encoder_lstm_nodes[i].h_t
							//#pragma omp parallel for
							for( int i = 0; i < current_minibatch_size; ++i ){
								for( int j = 0; j < h_s_size; ++j ){
									ctx_t.col(i) += ctx_alpha(j,i) * h_s_all[j].col(i);
								}
							}

							//r_h_t = W_h_to_atten_node.param * h_t
							W_h_to_atten_node.param->fProp(h_t, W_h_to_atten_node.fProp_matrix);

							//r_ctx_t = W_ctx_to_atten_node.param * ctx_t
							W_ctx_to_atten_node.param->fProp(ctx_t, W_ctx_to_atten_node.fProp_matrix);

							//atten_input_matrix = r_h_t + r_ctx_t
							atten_t_input_matrix.noalias() = W_h_to_atten_node.fProp_matrix + W_ctx_to_atten_node.fProp_matrix;

							//atten_t = tahn(atten_input_matrix+bias) performed by atten_t_node
							atten_t_node.param->fProp(atten_t_input_matrix, atten_t_node.fProp_matrix); //h_new_t = atten_t_node.fProp_matrx

							h_new_t = atten_t_node.fProp_matrix;
						}

					//attention node backpropagation: derivative to attention hidden state is input, output derivatives to h_t and h_s_all
					template<typename Derived>
						void bProp(const MatrixBase<Derived> &d_Err_t_to_d_h_new_t, const MatrixBase<Derived> &h_t, const vector<Eigen::Matrix<precision_type,Eigen::Dynamic, Eigen::Dynamic> > &h_s_all, const vector<int> &sequence_token_num, const unsigned int &max_input_sent_len) { 
							Matrix<precision_type,Dynamic,Dynamic> dummy_matrix;
							int current_minibatch_size = sequence_token_num.size();
							unsigned int h_s_size = max_input_sent_len;
							d_Err_t_to_d_h_s_all.resize(h_s_size);
							d_Err_t_to_d_alpha.setZero(h_s_size, minibatch_size);
							d_Err_t_to_d_sim.setZero(h_s_size, minibatch_size);
							for( int i = 0; i < h_s_size; ++i ){
								d_Err_t_to_d_h_s_all[i].resize(h_s_all[i].rows(), minibatch_size);
								d_Err_t_to_d_h_s_all[i].setZero(h_s_all[i].rows(), minibatch_size);
							}

							//d_h_t / atten_input_matrix = 1 - h_t^2
							atten_t_node.param->bProp(d_Err_t_to_d_h_new_t, atten_t_node.bProp_matrix, dummy_matrix, atten_t_node.fProp_matrix);
							atten_t_node.param->updateGradient(atten_t_node.bProp_matrix.leftCols(current_minibatch_size));

							//atten_input_matrix / h_t = W_h
							W_h_to_atten_node.param->bProp(atten_t_node.bProp_matrix, W_h_to_atten_node.bProp_matrix);

							//atten_input_matrix / W_h = h_t
							W_h_to_atten_node.param->updateGradient(atten_t_node.bProp_matrix.leftCols(current_minibatch_size), h_t.leftCols(current_minibatch_size));
							//atten_input_matrix / ctx_t = W_ctx
							W_ctx_to_atten_node.param->bProp(atten_t_node.bProp_matrix, W_ctx_to_atten_node.bProp_matrix);

							//atten_input_matrix / W_ctx = ctx_t
							W_ctx_to_atten_node.param->updateGradient(atten_t_node.bProp_matrix.leftCols(current_minibatch_size), ctx_t.leftCols(current_minibatch_size));

							//ctx_t / h_s_i = ctx_alpha_i
							#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){
								//d_Err_t_to_d_h_s_all[i] dim: hidden_size * minibatch_size
								//W_ctx_to_atten_node.bProp_matrix dim: hidden_size * minibatch_size
								//ctx_alpha dim: h_s_size * minibatch_size
								Matrix<precision_type,Dynamic,Dynamic> &tmp_matrix = d_Err_t_to_d_h_s_all[i];
								for( int j = 0; j < current_minibatch_size; ++j ){ 
									d_Err_t_to_d_h_s_all[i].col(j) = ctx_alpha(i,j) * W_ctx_to_atten_node.bProp_matrix.col(j);
								}
							}	

							//ctx_t / ctx_alpha_i = h_s_i
							//if h_s_all[i].col(j) = 0 then d_Err_t_to_d_alpha(i,j) = 0
							#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){
								//W_ctx_to_atten_node.bProp_matrix dim: hidden_size * minibatch_size
								//encoder_lstm_nodes[i].h_t dim: hidden_size * minibtach_size
								//d_Err_t_to_d_alpha dim: h_s_size * minibatch_size
								for( int j = 0; j < current_minibatch_size; ++j){
									if( i < h_s_size-sequence_token_num[j] )
										d_Err_t_to_d_alpha(i,j) = 0.;
									else
										d_Err_t_to_d_alpha(i,j) = W_ctx_to_atten_node.bProp_matrix.col(j).dot(h_s_all[i].col(j));
								}
							}

							//ctx_alpha_i / sim_scores_i = ctx_alpha_i*(1-ctx_alpha_i)
							//ctx_alpha_i / sim_scores_j = -ctx_alpha_i*ctx_alpha_j 
							//if ctx_alpha(i,k) = 0 then d_Err_t_to_d_sim(i,k) = 0
							//#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){
								//d_Err_t_to_d_alpha dim: h_s_size * minibatch_size
								//ctx_alpha dim: h_s_size * minibatch_size
								//d_Err_t_to_d_sim dim: h_s_size* minibatch_size
								for( int j = 0; j < h_s_size; ++j){
									for( int k = 0; k < current_minibatch_size; ++k ){
										d_Err_t_to_d_sim(j,k) += -1.0 * d_Err_t_to_d_alpha(i,k) * ctx_alpha(i,k) * ctx_alpha(j,k); 
									}
								}

								for( int k = 0; k < current_minibatch_size; ++k){
									d_Err_t_to_d_sim(i,k) += d_Err_t_to_d_alpha(i,k) * ctx_alpha(i,k); 
								}
							}

							//sim_scores_i / h_s_i = h_t
							//if d_Err_t_to_d_sim(i,j) = 0 then d_Err_t_to_d_h_s_all[i].col(j) = 0
							//#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){
								//d_Err_t_to_d_sim dim: h_s_size * minibatch_size
								//h_t dim: hidden_size * minibatch_size
								//d_Err_t_to_d_h_s_all[i] dim:: hidden_size * minibatch_size
								Matrix<precision_type,Dynamic,Dynamic> &tmp_matrix = d_Err_t_to_d_h_s_all[i];
								for( int j = 0; j < current_minibatch_size; ++j){
									d_Err_t_to_d_h_s_all[i].col(j) +=  h_t.col(j) * d_Err_t_to_d_sim(i,j);
								}
							}

							d_Err_t_to_d_h_t = W_h_to_atten_node.bProp_matrix;
							//sim_scores_i / h_t = h_s_i
							//if d_Err_t_to_d_sim(i,j) = 0 then d_Err_t_to_d_h_t.col(j) = 0
							//#pragma omp parallel for
							for( int i = 0; i < h_s_size; ++i ){
								//d_Err_t_to_dsim dim: h_s_size * minibatch_size
								//h_s_i dim: hidden_size * minibatch_size
								//d_Err_t_to_d_h_t dim: hidden_size * minibatch_size
								for( int j = 0; j < current_minibatch_size; ++j){
									d_Err_t_to_d_h_t.col(j) += h_s_all[i].col(j) * d_Err_t_to_d_sim(i,j);
								}
							}

						}

					//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block/
					//It also takes a list that says where the new hidden states must be obtained from. This is very useful for beam search.
					//In beam search, the hidden states must come from the current k-best sequence. The current k-best sequence might be continuations
					//of the same item or different items in the previous k-best sequence and thus we can't just copy over the previous hidden states
					//because some of the k-best sequences corresponding to previous hidden states might have died. 
					//TODO: should the indices be a vector ? 
					template <typename DerivedH, typename indexType>
						static void copyKBestHiddenStates(const MatrixBase<DerivedH> &from_h_matrix,
								const MatrixBase<DerivedH> &const_to_h_matrix,
								const vector<indexType> &from_indices) {
							int current_minibatch_size = from_indices.size();

							UNCONST(DerivedH, const_to_h_matrix, to_h_matrix);

							#pragma omp parallel for 
							for (int index=0; index<current_minibatch_size; index++){ 
								to_h_matrix.col(index) = from_h_matrix.col(from_indices.at(index)); 			
							}	
						}
				};

			class Standard_input_node{
				int minibatch_size;
				public:
				//Each LSTM node has a bunch of nodes and temporary data structures
				Node<Input_word_embeddings> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;

				Standard_input_node():
					minibatch_size(0),
					W_x_to_i_node(),
					W_x_to_f_node(),
					W_x_to_c_node(),
					W_x_to_o_node() {}	

				Standard_input_node(standard_input_model &input, int minibatch_size): 
					W_x_to_i_node(&input.W_x_to_i, minibatch_size),
					W_x_to_f_node(&input.W_x_to_f, minibatch_size),
					W_x_to_c_node(&input.W_x_to_c, minibatch_size),
					W_x_to_o_node(&input.W_x_to_o, minibatch_size),
					minibatch_size(minibatch_size) {
					}

				//Resizing all the parameters
				void resize(int minibatch_size){
					this->minibatch_size = minibatch_size;
					W_x_to_i_node.resize(minibatch_size);
					W_x_to_f_node.resize(minibatch_size);
					W_x_to_c_node.resize(minibatch_size);
					W_x_to_o_node.resize(minibatch_size);
				}
				template<typename Derived, typename DropMask>
					void fPropInputDropout(const MatrixBase<Derived> &data,
							const MatrixBase<DropMask> &dropout_mask) {
					}	

				template <typename Derived>
					void fProp(const MatrixBase<Derived> &data){
						W_x_to_c_node.param->fProp(data,W_x_to_c_node.fProp_matrix);
						W_x_to_f_node.param->fProp(data,W_x_to_f_node.fProp_matrix);
						W_x_to_o_node.param->fProp(data,W_x_to_o_node.fProp_matrix);
						W_x_to_i_node.param->fProp(data,W_x_to_i_node.fProp_matrix);			
					}	

				template<typename DerivedData, typename DerivedDIn>
					void bProp(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix){
						int current_minibatch_size = data.cols();
						W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data);
						W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data);
						W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data);	
						W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data);			
					}
			};

			class Google_input_node{
				int minibatch_size;
				public:
				//Each LSTM node has a bunch of nodes and temporary data structures
				Node<Input_word_embeddings> input_layer_node; //word id --> word embedding
				Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node; //word embedding --> forward output for different gates
				Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_x_t; //derivative to word embeddings
				Dropout_layer x_t_dropout_layer;
				Google_input_node():
					minibatch_size(0),
					input_layer_node(),
					W_x_to_i_node(),
					W_x_to_f_node(),
					W_x_to_c_node(),
					W_x_to_o_node() ,
					d_Err_t_to_n_d_x_t(Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic>()),
					x_t_dropout_layer(Dropout_layer()){}	

				//initialize the input node with the same model params
				Google_input_node(google_input_model &input, int minibatch_size): 
					input_layer_node(&input.input_layer, minibatch_size),
					W_x_to_i_node(&input.W_x_to_i, minibatch_size),
					W_x_to_f_node(&input.W_x_to_f, minibatch_size),
					W_x_to_c_node(&input.W_x_to_c, minibatch_size),
					W_x_to_o_node(&input.W_x_to_o, minibatch_size),
					x_t_dropout_layer(Dropout_layer()),
					minibatch_size(minibatch_size) {
					}

				//Resizing all the parameters
				void resize(int minibatch_size,
						double dropout_probability){
					this->minibatch_size = minibatch_size;
					input_layer_node.resize(minibatch_size);
					W_x_to_i_node.resize(minibatch_size);
					W_x_to_f_node.resize(minibatch_size);
					W_x_to_c_node.resize(minibatch_size);
					W_x_to_o_node.resize(minibatch_size);
					d_Err_t_to_n_d_x_t.resize(input_layer_node.param->n_outputs(),minibatch_size);
				}

				void resizeDropout(int minibatch_size,
						precision_type dropout_probability){
					resize (minibatch_size);
					//If using dropout
					if (dropout_probability > 0) {
						x_t_dropout_layer = Dropout_layer(input_layer_node.param->n_outputs(), 
								minibatch_size, 
								1.-dropout_probability);
					} else {
						cerr<<"The dropout probability has to be greater than 0!"<<endl;
						exit(1);
					}			
				}

				//Resizing all the parameters
				void resize(int minibatch_size){
					this->minibatch_size = minibatch_size;
					input_layer_node.resize(minibatch_size);
					W_x_to_i_node.resize(minibatch_size);
					W_x_to_f_node.resize(minibatch_size);
					W_x_to_c_node.resize(minibatch_size);
					W_x_to_o_node.resize(minibatch_size);
					d_Err_t_to_n_d_x_t.resize(input_layer_node.param->n_outputs(),minibatch_size);
				}


				template <typename Derived>
					void fProp(const MatrixBase<Derived> &data){
						fPropInput(data);
						fPropProjections();
					}

				template <typename Derived, typename Engine>
					void fPropDropout(const MatrixBase<Derived> &data,
							Engine &eng){
						fPropInput(data);
						x_t_dropout_layer.fProp(input_layer_node.fProp_matrix,eng);
						fPropProjections();
					}


				//word id --> word embedding
				template <typename Derived>
					void fPropInput(const MatrixBase<Derived> &data){
						input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
					}


				//word embedding --> forward outputs for different gates
				void fPropProjections(){
					W_x_to_c_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_c_node.fProp_matrix);
					W_x_to_f_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_f_node.fProp_matrix);
					W_x_to_o_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_o_node.fProp_matrix);
					W_x_to_i_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_i_node.fProp_matrix);	
				}

				template<typename DerivedData, typename DerivedDIn>
					void bProp(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix){
						int current_minibatch_size = data.cols();
						bPropProjections(o_t_node_bProp_matrix,
								i_t_node_bProp_matrix,
								f_t_node_bProp_matrix,
								tanh_c_prime_t_node_bProp_matrix,
								current_minibatch_size);
						bPropInput(data);
					}

				template<typename DerivedData, typename DerivedDIn>
					void bPropDropout(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix) {
						int current_minibatch_size = data.cols();			
						bPropProjections(o_t_node_bProp_matrix,
								i_t_node_bProp_matrix,
								f_t_node_bProp_matrix,
								tanh_c_prime_t_node_bProp_matrix,
								current_minibatch_size);
						x_t_dropout_layer.bProp(d_Err_t_to_n_d_x_t);
						bPropInput(data);				
					}


				template <typename DerivedDIn>
					void bPropProjections(const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix,
							int current_minibatch_size){
						W_x_to_c_node.param->bProp(tanh_c_prime_t_node_bProp_matrix,
								W_x_to_c_node.bProp_matrix);
						W_x_to_o_node.param->bProp(o_t_node_bProp_matrix,
								W_x_to_o_node.bProp_matrix);
						W_x_to_f_node.param->bProp(f_t_node_bProp_matrix,
								W_x_to_f_node.bProp_matrix);
						W_x_to_i_node.param->bProp(i_t_node_bProp_matrix,
								W_x_to_i_node.bProp_matrix);

						W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix.leftCols(current_minibatch_size),
								input_layer_node.fProp_matrix.leftCols(current_minibatch_size));
						W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix.leftCols(current_minibatch_size),
								input_layer_node.fProp_matrix.leftCols(current_minibatch_size));
						W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix.leftCols(current_minibatch_size),
								input_layer_node.fProp_matrix.leftCols(current_minibatch_size));	
						W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix.leftCols(current_minibatch_size),
								input_layer_node.fProp_matrix.leftCols(current_minibatch_size));		
						d_Err_t_to_n_d_x_t = W_x_to_c_node.bProp_matrix + 
							W_x_to_o_node.bProp_matrix +
							W_x_to_f_node.bProp_matrix +
							W_x_to_i_node.bProp_matrix;												
					}

				//update word embeddings
				template<typename DerivedData>
					void bPropInput(const MatrixBase<DerivedData> &data) {
						int current_minibatch_size = data.cols();
						input_layer_node.param->updateGradient(d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size),
								data);				
					}
			};

			//Hidden_to_hidden_input_node used for attention node to link attention output to next LSTM node
			//It can also be used to link between different layers (e.g. bottom encoder layer --> upper encoder layer if we design stack layers)
			//It is similar to google_input_node and the difference is that the input of Hidden_to_hidden_input_node is attention output embeddings rather than word id
			class Hidden_to_hidden_input_node{
				int minibatch_size;
				precision_type norm_threshold;
				public:
				//Each LSTM node has a bunch of nodes and temporary data structures
				Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
				Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_x_t;

				Hidden_to_hidden_input_node():
					minibatch_size(0),
					W_x_to_i_node(),
					W_x_to_f_node(),
					W_x_to_c_node(), 
					W_x_to_o_node() {}	

				Hidden_to_hidden_input_node(hidden_to_hidden_input_model &input, int minibatch_size): 
					W_x_to_i_node(&input.W_x_to_i, minibatch_size),
					W_x_to_f_node(&input.W_x_to_f, minibatch_size),
					W_x_to_c_node(&input.W_x_to_c, minibatch_size),
					W_x_to_o_node(&input.W_x_to_o, minibatch_size),
					minibatch_size(minibatch_size), 
					norm_threshold(input.norm_threshold) {
					}

				//Resizing all the parameters
				void resize(int minibatch_size){
					this->minibatch_size = minibatch_size;
					W_x_to_i_node.resize(minibatch_size);
					W_x_to_f_node.resize(minibatch_size);
					W_x_to_c_node.resize(minibatch_size);
					W_x_to_o_node.resize(minibatch_size);
					d_Err_t_to_n_d_x_t.resize(W_x_to_c_node.param->n_inputs(),minibatch_size);
				}

				template<typename Derived, typename DropMask>
					void fPropInputDropout(const MatrixBase<Derived> &data,
							const MatrixBase<DropMask> &dropout_mask) {
					}	

				template <typename Derived>
					void fProp(const MatrixBase<Derived> &data){
						W_x_to_c_node.param->fProp(data,W_x_to_c_node.fProp_matrix);
						W_x_to_f_node.param->fProp(data,W_x_to_f_node.fProp_matrix);
						W_x_to_o_node.param->fProp(data,W_x_to_o_node.fProp_matrix);
						W_x_to_i_node.param->fProp(data,W_x_to_i_node.fProp_matrix);				
					}	

				template<typename DerivedData, typename DerivedDIn>
					void bProp(const MatrixBase<DerivedData> &data,
							const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
							const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix,
							const int &current_minibatch_size){
						W_x_to_c_node.param->bProp(tanh_c_prime_t_node_bProp_matrix,
								W_x_to_c_node.bProp_matrix);
						W_x_to_o_node.param->bProp(o_t_node_bProp_matrix,
								W_x_to_o_node.bProp_matrix);
						W_x_to_f_node.param->bProp(f_t_node_bProp_matrix,
								W_x_to_f_node.bProp_matrix);
						W_x_to_i_node.param->bProp(i_t_node_bProp_matrix,
								W_x_to_i_node.bProp_matrix);

						W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data.leftCols(current_minibatch_size));
						W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data.leftCols(current_minibatch_size));
						W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data.leftCols(current_minibatch_size));	
						W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix.leftCols(current_minibatch_size),
								data.leftCols(current_minibatch_size));		

						d_Err_t_to_n_d_x_t = W_x_to_c_node.bProp_matrix +
							W_x_to_o_node.bProp_matrix +
							W_x_to_f_node.bProp_matrix +
							W_x_to_i_node.bProp_matrix;	
						//scaleAndNormClip(d_Err_t_to_n_d_x_t, current_minibatch_size, norm_threshold);
					}
			};

		} // namespace nplm
#undef EIGEN_NO_AUTOMATIC_RESIZING
