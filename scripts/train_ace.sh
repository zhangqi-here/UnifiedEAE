if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/rams_VIB_AVG_weight_time/$SEED/$LR
mkdir -p $work_path

python engine.py \
    --model_type=paie \
    --dataset_type=ace_wiki \
    --model_name_or_path=facebook/bart-base \
    --role_path=./data/dset_meta/description_ace.csv#./data/dset_meta/description_wikievent.csv#./data/dset_meta/description_rams.csv \
    --prompt_path=./data/prompts/prompts_ace_full.csv#./data/prompts/prompts_wikievent_full.csv#./data/prompts/prompts_rams_full.csv \
    --seed=$SEED \
    --output_dir=$work_path  \
    --learning_rate=$LR \
    --batch_size 4 \
    --eval_steps 500  \
    --max_steps=15000 \
    --max_enc_seq_lengt 500 \
    --max_prompt_seq_length 80 \
    --bipartite \
    --inference_only \
    --inference_model_path=/workspace/PAIE-main_VIB/exps/ace05_wiki_VIB_AVG_weight_auto_beta-5_25000/42/2e-5/checkpoint
