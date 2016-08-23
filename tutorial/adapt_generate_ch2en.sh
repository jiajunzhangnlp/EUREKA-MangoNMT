#! /bin/bash

../src-full-open/generateFromNetworkAdapt --beam_size 10 --num_noise_samples 500 --loss_function nce --learning_rate 0.01 --training_sent_file data/spoken.train --num_similar_samples 10 --similarity_threshold 0.3 --norm_threshold 1.0 --min_output_ratio 0.8 --max_output_ratio 2.4 --decoder_model_file $1 --encoder_model_file $2 --encoder_reverse_model_file $3 --testing_sent_file data/spoken.test --predicted_sequence_file spoken.test.out
