model=transformer
PROBLEM=wmt14_enfr
SETTING=transformer_big

mkdir -p $model/$PROBLEM/$SETTING

python train.py data-bin/wmt14_en_fr \
	--clip-norm 0.1 --dropout 0.1 --max-tokens 3072 \
	--arch $SETTING --save-dir $model/$PROBLEM/$SETTING \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --lr 0.5 --optimizer nag --warmup-init-lr 0.5 \
	--warmup-updates 16000 --max-update 300000 --save-interval 10000

