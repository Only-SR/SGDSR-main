
python main.py --n_hid 64 --dataset yelp --n_layers 2 --s_layer 2 --n_lay 2 --s_lay 2  --lr1 0.00025 --lr2 0.00025 --lr 0.001 \
 --difflr1 0.0001 --difflr2 0.0001 --difflr 0.0001 --reg1 0.0001 --reg2 0.0001 --reg 0.001 --batch_size 2560 --test_batch_size 2048 \
  --emb_size 16 --steps 20 --noise_scale 0.1 --model_dir './Model/yelp/'  --tau 0.8 --n_epoch 800