python main.py --n_hid 80 --dataset ciao --n_layers 1 --s_layer 1 --n_lay 2 --s_lay 2  --lr1 0.005 --lr2 0.005 --lr 0.005 --decay_step 4 \
 --difflr1 0.001 --difflr2 0.001 --difflr 0.001 --reg1 0.01 --reg2 0.01 --reg 0.01 --batch_size 2560 --test_batch_size 2048 \
  --emb_size 16 --steps 20 --noise_scale 1 --model_dir './Model/ciao/'  --n_epoch 800 --tau 0.9