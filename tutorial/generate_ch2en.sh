#! /bin/bash

../src/generateFromNetwork --num_threads 24 --beam_size 10 --decoder_model_file $1 --encoder_model_file $2 --encoder_reverse_model_file $3 --testing_sent_file data/spoken.test --predicted_sequence_file spoken.test.out
