

for roi_idx in {0..144}; do
  CUDA_VISIBLE_DEVICES=1 python dkgp.py \
    --file "$file" \
    --kfoldID "$kfoldID" \
    --experimentID "$experimentID" \
    --covariates "$covariates" \
    --iterations "$iterations" \
    --personalization False \
    --roi_idx "$roi_idx" \
    --folder "$folder" \
    --learning_rate 0.02
done