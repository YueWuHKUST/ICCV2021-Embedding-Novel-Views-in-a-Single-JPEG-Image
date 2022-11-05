OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python render_test.py \
    --root_dir ./checkpoints/stereo_final/test_result/ \
    --rotation_render_max_multiples 0 \
    --output_dir checkpoints/stereo_final/render \
    --save_img
#/disk1/guotao/code/stereo-magnification_hide_mpi_pytorch/checkpoints/ours/test_result