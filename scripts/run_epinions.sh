
python main.py --n_hid 64 --dataset epinions --n_layers 2 --s_layer 2 --n_lay 2 --s_lay 2  --lr1 0.0003 --lr2 0.0003 --lr 0.005 \
 --difflr1 0.001 --difflr2 0.001 --difflr 0.001 --reg1 0.00005 --reg2 0.00005 --reg 0.001 --batch_size 2560 --test_batch_size 1024 \
  --emb_size 16 --steps 150 --noise_scale 0.05 --model_dir './Model/epinions/'  --n_epoch 800 --tau 0.8