import argparse
import time

def get_opts():
    flags = argparse.ArgumentParser(description='Argument for hide mpi')

    auto_name='mpi_'+'_'.join(time.strftime("%b %d %l-%M%p").split())
    # general
    flags.add_argument('--name', type=str, default=auto_name, help='Name for the experiment to run.')

    # networt
    flags.add_argument('--encoder_type', type=int, default=1,help='select the type of encoder. 1 for Encoder, 2 for EncoderAlphaWeight, 3 for UnetEncoder, 4 for ResUnetEncoder')
    flags.add_argument('--decoder_type', type=int, default=1,help='select the type of encoder. 1 for Decoder, 2 for DecoderDepthwise, 3 for UnetDecoder, 4 for ResUnetDecoder')

    # loss
    flags.add_argument('--lambda_l2_mpi_rgb', type=float, default=1.0,help='weight for l2 loss of the rgb in mpis')
    flags.add_argument('--lambda_l2_mpi_alpha', type=float, default=1.0,help='weight for l2 loss of the alpha in mpis')
    flags.add_argument('--lambda_vgg_ref', type=float, default=1.0,help='weight for vgg loss of embedding image')
    flags.add_argument('--lambda_l2_ref', type=float, default=1.0,help='weight for l2 loss of embedding image')
    flags.add_argument('--lambda_discriminator', type=float, default=1.0,help='weight for the GAN loss')
    flags.add_argument('--lambda_l2_render_loss', type=float, default=1.0,help='weight for l2 loss of the random render')
    flags.add_argument('--lambda_vgg_render_loss', type=float, default=1.0,help='weight for vgg loss of the random render')
    flags.add_argument('--max_translation', type=float, default=0.5,help='max translation of x y z axis')
    flags.add_argument('--max_rotation', type=float, default=8,help='max rotation angle of x y z axis, in degree')
    flags.add_argument('--fx', type=float, default=0.5,help='fx in camera matrix')
    flags.add_argument('--fy', type=float, default=0.5,help='fy in camera matrix')
    flags.add_argument('--lambda_fft', type=float, default=1.0,help='weight for l2 loss of the ref on frequency domain')
    flags.add_argument('--lambda_reg', type=float, default=1.0,help='weight for regularization')

    flags.add_argument('--use_reg', action='store_true',help='If in training phase')

    # data
    flags.add_argument('--num_workers', type=int, default=4, help='Number of MPI planes')
    flags.add_argument('--image_dir', type=str, default='images', help='Path to training image directories.')
    flags.add_argument('--batch_size_each_gpu', type=int, default=1, help='The size of a sample batch.')
    flags.add_argument('--mpi_num', type=int, default=32, help='Number of MPI planes')
    flags.add_argument('--image_height', type=int, default=256, help='Image height in pixels.')
    flags.add_argument('--image_width', type=int, default=256, help='Image width in pixels.')
    flags.add_argument('--random_resize_ratio', type=float, default=0.2, help='randomly resize the image in this ratio')
    flags.add_argument('--encode_diff', action='store_true',help='only_give the difference between rgba layers to the reference as input to the network encoder')
    flags.add_argument('--isTrain', action='store_true',help='If in training phase')

    
    flags.add_argument('--load_path_npy', action='store_true',help='whether load saved path npy')
    flags.add_argument('--path_npy', type=str, default=None, help='presaved path npy file')

    # optmizer
    flags.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    flags.add_argument('--decay_step', nargs='+', type=int, default=[20],help='scheduler decay step')
    flags.add_argument('--decay_gamma', type=float, default=0.5,help='learning rate decay amount')

    # display
    flags.add_argument('--print_freq', type=int, default=100, help='Logging frequency.')
    flags.add_argument('--display_freq', type=int, default=2000, help='Logging frequency.')

    # training
    flags.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps, -1 for unlimited')
    flags.add_argument('--max_epoches', type=int, default=-1, help='Maximum number of training epoches, -1 for unlimited')
    flags.add_argument('--continue_train', action='store_true',help='Continue training from previous checkpoint.')
    flags.add_argument('--save_training_results', action='store_true',help='Save the results during training')
    flags.add_argument('--start_epoch', type=int, default=1, help='the epoch index to start or continue training')
    flags.add_argument('--start_step', type=int, default=1, help='the step index to start or continue training')
    flags.add_argument('--encoder_ckpt_path', type=str, default=None, help='Location to the saved encoder model.')
    flags.add_argument('--decoder_ckpt_path', type=str, default=None, help='Location to the saved decoder model.')
    flags.add_argument('--discriminator_ckpt_path', type=str, default=None, help='Location to the saved discriminator model.')

    ## load previous model
    flags.add_argument('--load_pretrain', type=str, default=None, help='load pretrained model path')
    flags.add_argument('--which_epoch', type=int, default=None, help='load epoch')
    flags.add_argument('--which_iter', type=int, default=None, help='load iter')


    # check point saving
    flags.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Location to save the models.')
    flags.add_argument('--model_save_freq', type=int, default=2000, help='Frequency of sving the model')

    # jpeg
    flags.add_argument('--jpeg_compress', action='store_true',help='If use differentiable jpeg compression')
    flags.add_argument('--jpeg_quality', type=int, default=90, help='the jpeg quality')
    flags.add_argument('--jpeg_quality_range', type=int, default=3, help='the jpeg quality range, random pick from jpeg_quality-range to jpeg_quality+range')

    # quantization
    flags.add_argument('--quantize', action='store_true',help='If use differentiable quantization')

    # color jitter
    flags.add_argument('--color_jitter', action='store_true',help='If use color_jittering')
    flags.add_argument('--hue', type=float, default=0.1,help='the range of the random hue adjusting')
    flags.add_argument('--brightness', type=float, default=0.2,help='the range of the random brightness adjusting')
    flags.add_argument('--saturation', type=float, default=0.5,help='the range of the random saturation adjusting')
    flags.add_argument('--contrast', type=float, default=0.2,help='the range of the random contrast adjusting')

    # random crop resize
    flags.add_argument('--random_crop_resize', action='store_true',help='If use color_jittering')
    flags.add_argument('--min_crop_size', type=float, default=0.8,help='the min crop ratio')
    flags.add_argument('--min_resize_size', type=float, default=0.8,help='the min resize ratio')
    flags.add_argument('--max_resize_size', type=float, default=1.2,help='the max resize ratio')

    # test
    flags.add_argument('--output_dir', type=str, default='output', help='Location to save the test result.')


    # dataset
    flags.add_argument('--dataset', type=str, default='stereo', help='choose stereo or pb')
    flags.add_argument('--feat_num', type=int, default=64, help='feat number')


    return flags.parse_args()
