#! /bin/bash


../src/trainNeuralNetwork --use_feed_input 1 --dropout_probability 0.2 --num_threads 20 --training_sent_file data/spoken.train --validation_sent_file data/spoken.valid --input_vocab_size 4000 --output_vocab_size 4000 --learning_rate 1 --norm_threshold 1.0 --num_hidden 64 --model_prefix lstm.s2s.z2e --minibatch_size 32 --validation_minibatch_size 10 --init_range 0.1 --num_epochs 20 --loss_function nce --num_noise_samples 500
