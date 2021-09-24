# model 2 lending club
#THEANO_FLAGS=device=cuda0,floatX=float32 python -u training_model2.py --outfile m2 --nlayers_enc 3 --hdim_enc 10 40 10 --nlayers_dec 3 --hdim_dec 10 40 10 --nlayers_cls 1 --hdim_cls 130 --nlayers_aux_inf 2 --hdim_aux_inf 10 40 --zdim 50 --adim 50 --n_sup 1552 --n_unsup 30997 --epochs 501 --n_cv 5 --pre_epochs 10 --pre_auc 0.5 --beta 0.008 --at_epoch 50 --threshold 0.62 2>&1 | tee -a ../../output/log.txt
THEANO_FLAGS=device=cuda0,floatX=float32 python -u test_model2.py
