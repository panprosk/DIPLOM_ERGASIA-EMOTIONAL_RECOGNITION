#!/bin/bash
cd /scratch/st1084601/thesis
source venv/bin/activate
run() {
  name=$1; shift
  python train_hybrid_baseline.py --wandb --wandb-project deap-hybrid-cnn-mlp --wandb-run-name "v2-$name" \
    --seed 42 --num-workers 2 --checkpoint-name "v2_$name.pt" "$@" > "outputs/v2_$name.log" 2>&1
}
lane1() {
  run reg_d50_wd1e2 --dropout 0.5 --weight-decay 1e-2
  run lowlr_wd1e2 --lr 1e-4 --weight-decay 1e-2 --dropout 0.4
  run emb64_d50 --embedding-dim 64 --dropout 0.5 --weight-decay 5e-3
  run ls10_mix4 --label-smoothing 0.1 --mixup-alpha 0.4 --dropout 0.4
}
lane2() {
  run adv --enable-adversarial
  run adv_reg --enable-adversarial --dropout 0.5 --weight-decay 1e-2
  run contrastive --enable-contrastive
  run adv_contrastive --enable-adversarial --enable-contrastive --dropout 0.4
}
lane3() {
  run lowlr_adv --lr 1e-4 --enable-adversarial --weight-decay 1e-2 --dropout 0.4
  run no_gate_balance --gate-balance-weight 0 --dropout 0.4 --weight-decay 5e-3
  run noeeg_feat --no-eeg-handcrafted --dropout 0.4 --weight-decay 5e-3
  run bs256_lr5e4 --batch-size 256 --lr 5e-4 --dropout 0.4 --weight-decay 5e-3
}
lane1 & sleep 5; lane2 & sleep 5; lane3 &
wait
