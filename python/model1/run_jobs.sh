# define inputs for each job that will be run
#THEANO_FLAGS=device=cuda0,floatX=float32 python -u training_model1.py --outfile m1 --epochs 401 --n_cv 3 --beta 1.1 --dset paper --n_sup 1552 --n_unsup 30997 
THEANO_FLAGS=device=cuda0,floatX=float32 python -u test_model1.py 
