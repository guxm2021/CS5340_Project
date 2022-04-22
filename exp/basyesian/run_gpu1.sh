for n_sghmc in 4 6 8 10 12 14
do
  for alpha in 0.1 0.5 0.8 0.01 0.05 0.08
  do 
    for lambda_noise in 0.1 0.5 0.01 0.05 0.001 0.005
    do
      echo "The value of n_sghmc is $n_sghmc and the value of alpha is $alpha and the value of lambda_noise is $lambda_noise!"
      python run.py --gpu 1 --model probGRU --lr 5e-3 --quantization 0.1 --seed 2233 --bayes --n_sghmc $n_sghmc --alpha $alpha --lambda_noise $lambda_noise
    done
  done
done