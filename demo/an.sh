python3 inference_tse_v2.py \
  --mixture_wav demo/input/multispeaker/000000_s1-libri_2094_s2-libri_7127_ov80.mix.wav \
  --enroll_wav demo/input/multispeaker/2094-142345-0059.wav \
  --target_age adult \
  --reference_dir demo/reference_audio \
  --out_dir demo/output/multispeaker/ov80/ \
  --t_start_sec 0 \
  --duration 5 \
  --recombine residual_add \
  --min_duration 1.0



python3 inference_tse_v2.py \
  --mixture_wav demo/input/multispeaker/000001_s1-myst_999455_s2-libri_7729_ov40.mix.wav \
  --enroll_wav  demo/input/multispeaker/myst_999455_2009-07-12_00-00-00_MS_2.1_002.wav\
  --target_age child \
  --reference_dir demo/reference_audio \
  --out_dir demo/output/multispeaker/myst-ac/ \
  --t_start_sec 0 \
  --duration 5 \
  --recombine residual_add \
  --min_duration 1.0



