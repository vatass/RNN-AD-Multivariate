#!/bin/bash

file="subjectsamples_longclean_dl_muse_allstudies"
kfoldID="missingdasae"
experimentID="allstudies"
covariates="Diagnosis-Age-Sex-APOE4_Alleles-Education_Years"
iterations=200
folder=1


for roi_idx in {0..144}; do
  CUDA_VISIBLE_DEVICES=1 python dkgp.py \
    --file "$file" \
    --kfoldID "$kfoldID" \
    --experimentID "$experimentID" \
    --covariates "$covariates" \
    --iterations "$iterations" \
    --personalization False \
    --roi_idx "$roi_idx" \
    --learning_rate 0.02
done