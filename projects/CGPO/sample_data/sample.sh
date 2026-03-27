OPENAI_API_KEY="sk-0057d2b0958d41588d9f61d4ae6c15a7" \
python projects/CGPO/sample_data/repeat_sample.py \
    --stream true \
    --sampler_type distill \
    --sampler_engine client \
    --engine_kwargs '{"base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"}' \
    --model qwen3-vl-plus-2025-12-19 \
    --dataset hf::PAPOGalaxy/PAPO_ViRL39K_train \
    --load_from_cache_file true \
    --remove_unused_columns false \
    --dataset_num_proc 64 \
    --num_return_sequences 1 \
    --max_retries 10 \
    --orm_model accuracy \
    --temperature 0.9 \
    --top_p 0.8 \
    --output_dir outputs/CGPO/sample/ViRL39K \
    --output_file right.json \
    --resume true
