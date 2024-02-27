#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
save_dir="/Users/jongbeomkim/Desktop/workspace/DDIM/samples/"
img_size=64
n_ddim_steps=50
eta=0
trunc_normal_thresh=2.5

python3 ../sample.py\
    --mode="normal"\
    --model_params="$model_params"\
    --save_path="$save_dir/normal/test.jpg"\
    --img_size=$img_size\
    --batch_size=1\
    --trunc_normal_thresh=$trunc_normal_thresh\

# python3 ../sample.py\
#     --mode="interpolation"\
#     --model_params="$model_params"\
#     --save_path="$save_dir/interpolation/4.jpg"\
#     --img_size=$img_size\
#     --trunc_normal_thresh=$trunc_normal_thresh\

# python3 ../sample.py\
#     --mode="interpolation_on_grid"\
#     --model_params="$model_params"\
#     --save_path="$save_dir/interpolation_on_grid/0.jpg"\
#     --img_size=$img_size\
#     --n_rows=10\
#     --n_cols=10\
#     --trunc_normal_thresh=$trunc_normal_thresh\
