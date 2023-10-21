CUDA_VISIBLE_DEVICES=7 python ./train_ref.py ./configs/thumos_i3d.yaml --output demo
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 25
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 30
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 35

CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 29
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 31


CUDA_VISIBLE_DEVICES=6 python ./train_ref.py ./configs/thumos_i3d.yaml --output demo1
CUDA_VISIBLE_DEVICES=6 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo1_ref -epoch 25
CUDA_VISIBLE_DEVICES=6 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo1_ref -epoch 30
CUDA_VISIBLE_DEVICES=6 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo1_ref -epoch 35

CUDA_VISIBLE_DEVICES=3 python ./train_ref.py ./configs/thumos_i3d.yaml --output demo2
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo2_ref -epoch 25
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo2_ref -epoch 30
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo2_ref -epoch 35

CUDA_VISIBLE_DEVICES=4 python ./train_ref.py ./configs/thumos_i3d.yaml --output demo3
CUDA_VISIBLE_DEVICES=4 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo3_ref -epoch 25
CUDA_VISIBLE_DEVICES=4 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo3_ref -epoch 30
CUDA_VISIBLE_DEVICES=4 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo3_ref -epoch 35

CUDA_VISIBLE_DEVICES=3 python ./train_ref.py ./configs/thumos_i3d.yaml --output sota
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_sota_ref -epoch 29
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_sota_ref -epoch 31
CUDA_VISIBLE_DEVICES=3 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_sota_ref -epoch 32


CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 26
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 27
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 28
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 29
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 31
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 32
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 33
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 34

CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 29
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 32
CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 34



CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 40


CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_demo_ref -epoch 31

CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_6785_ref -epoch 31

CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_6790_ref -epoch 23

2 1.5 -- 31 6785

==================================================================
CUDA_VISIBLE_DEVICES=6 python ./train_af.py ./configs/anet_tsp.yaml --output demo

CUDA_VISIBLE_DEVICES=6 python ./train_ref.py ./configs/anet_tsp.yaml --output demo

CUDA_VISIBLE_DEVICES=4 python ./eval_all.py ./configs/anet_tsp.yaml ./ckpt/anet_tsp_demo_ref -epoch 15
==================================================================
CUDA_VISIBLE_DEVICES=7 python train_ref.py thumos14  --cfg ${cfg_path} --snapshot_pref outputs/snapshots/${exp_name}/ --epochs ${max_epoch}



CUDA_VISIBLE_DEVICES=7 python ./eval_all.py ./configs/thumos_i3d.yaml ../ckpt/thumos_i3d_Thumos_ref -epoch 35

CUDA_VISIBLE_DEVICES=0 python ./train_ref.py ./configs/thumos_i3d.yaml --output Thumos