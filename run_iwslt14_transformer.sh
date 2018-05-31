model=transformer
PROBLEM=IWSLT14_DEEN
SETTING=transformer_small

mkdir -p $model/$PROBLEM/$SETTING

CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
	--clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
	--arch $SETTING --save-dir $model/$PROBLEM/$SETTING \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --lr 0.25 --optimizer nag --warmup-init-lr 0.25 \
	--warmup-updates 4000 --max-update 100000
