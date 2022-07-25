### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy
import functools
# from models.differentiable_quantize import DifferentiableQuantize

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)

def define_encoder(gpu_ids=[], opt=[]):
    net_encoder = Encoder()
    print_network(net_encoder)
    if len(gpu_ids) > 0:
        net_encoder.cuda(gpu_ids[0])
    net_encoder.apply(weights_init)
    return net_encoder

def define_decoder(gpu_ids=[], opt=[]):   
    net_decoder = Decoder()
    print_network(net_decoder)
    if len(gpu_ids) > 0:
        net_decoder.cuda(gpu_ids[0])
    net_decoder.apply(weights_init)
    return net_decoder

# Leave for potential GAN training
def define_D(input_nc, ndf, n_layers_D, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, num_D, getIntermFeat)   
    print_network(netD)
    if len(gpu_ids) > 0:    
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    #print('Total number of parameters: %d' % num_params)





class InputFeatureExtractor(nn.Module):
    def __init__(self,input_channel=4, feature_channel=64, layer_num=1):
        super(InputFeatureExtractor, self).__init__()
        layer_list=[]
        layer_list.append(nn.Conv2d(input_channel,feature_channel, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU())
        for _ in range(layer_num-1):
            layer_list.append(nn.Conv2d(feature_channel,feature_channel, kernel_size=3, stride=1, padding=1))
            layer_list.append(nn.ReLU())
        self.layers=nn.Sequential(*layer_list)
    def forward(self,x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self,feature_channel=64,layer_num=2):
        super(ResBlock,self).__init__()
        layer_list=[]
        for idx in range(layer_num):
            layer_list.append(nn.Conv2d(feature_channel,feature_channel, kernel_size=3, stride=1, padding=1))
            if idx<(layer_num-1):
                layer_list.append(nn.ReLU())
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        x_input=x
        x=self.layers(x)
        x=x+x_input
        return x


class DecoderResBlock(nn.Module):
    def __init__(self,feature_channel=64):
        super(DecoderResBlock,self).__init__()
        self.layers=nn.Sequential(nn.Conv2d(feature_channel,feature_channel*2, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(feature_channel*2,feature_channel, kernel_size=3, stride=1, padding=1))

    def forward(self,x):
        x_input=x
        x=self.layers(x)
        x=x+x_input
        return x


class Encoder(nn.Module):
    def __init__(self, input_num=32,
                    ref_channel=3,
                    input_channel=4,
                    output_channel=3,
                    input_feature_extractor_layers=1,
                    base_feature_channels=64,
                    start_resblock_num=2,
                    unet_resblock_num=4,
                    end_resblock_num=2
                    ):
        super(Encoder, self).__init__()
        self.input_num=input_num

        self.ref_feature_extractor=InputFeatureExtractor(ref_channel,base_feature_channels,input_feature_extractor_layers)

        self.input_feature_extractors=nn.ModuleList()
        self.input_weight_layers=nn.ModuleList()
        for _ in range(input_num):
            self.input_feature_extractors.append(InputFeatureExtractor(input_channel,base_feature_channels,input_feature_extractor_layers))
            self.input_weight_layers.append(nn.Conv2d(base_feature_channels,1, kernel_size=3, stride=1, padding=1))

        self.softmax=nn.Softmax(dim=1)

        self.ref_merge=nn.Sequential(
                                    nn.Conv2d(base_feature_channels*2,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())

        start_feature_res_blocks=[]
        for _ in range(start_resblock_num):
            start_feature_res_blocks.append(ResBlock(base_feature_channels,2))
        self.start_feature_res_blocks=nn.Sequential(*start_feature_res_blocks)

        self.unet_conv_1=nn.Sequential(
                                        nn.Conv2d(base_feature_channels,base_feature_channels*2, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        self.unet_conv_2=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*4, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*4,base_feature_channels*4, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        unet_feature_res_blocks=[]
        for _ in range(unet_resblock_num):
            unet_feature_res_blocks.append(ResBlock(base_feature_channels*4,2))
        self.unet_feature_res_blocks=nn.Sequential(*unet_feature_res_blocks)

        self.unet_conv_up_1=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*6,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        self.unet_conv_up_2=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*3,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        end_feature_res_blocks=[]
        for _ in range(end_resblock_num):
            end_feature_res_blocks.append(ResBlock(base_feature_channels,2))
        self.end_feature_res_blocks=nn.Sequential(*end_feature_res_blocks)

        self.output_conv=nn.Sequential(
                                    nn.Conv2d(base_feature_channels,output_channel, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())

    def forward(self, input_list):

        ref_feature=self.ref_feature_extractor(input_list[0])


        feature_list=[]
        weight_list=[]
        for input_idx,input_tensor in enumerate(input_list[1]):
            input_feature=self.input_feature_extractors[input_idx](input_tensor)
            input_weight=self.input_weight_layers[input_idx](input_feature)
            feature_list.append(input_feature.unsqueeze(1))
            weight_list.append(input_weight.unsqueeze(1))

        weight=torch.cat(weight_list,dim=1)
        weight=self.softmax(weight)

        feature=torch.cat(feature_list,dim=1)

        feature=feature*weight
        feature=torch.sum(feature,dim=1)

        feature=torch.cat([ref_feature,feature],dim=1)
        feature=self.ref_merge(feature)

        feature=self.start_feature_res_blocks(feature)

        feature_1=feature
        feature=self.unet_conv_1(feature)

        feature_2=feature
        feature=self.unet_conv_2(feature)

        feature=self.unet_feature_res_blocks(feature)

        feature=F.interpolate(feature,(feature_2.size(2),feature_2.size(3)))
        feature=torch.cat((feature,feature_2),dim=1)

        feature=self.unet_conv_up_1(feature)

        feature=F.interpolate(feature,(feature_1.size(2),feature_1.size(3)))
        feature=torch.cat((feature,feature_1),dim=1)

        feature=self.unet_conv_up_2(feature)

        feature=self.end_feature_res_blocks(feature)

        feature=self.output_conv(feature)

        return feature







class EncoderAlphaWeight(nn.Module):
    def __init__(self, input_num=32,
                    ref_channel=3,
                    input_rgb_channel=3,
                    input_alpha_channel=1,
                    output_channel=3,
                    input_feature_extractor_layers=1,
                    base_feature_channels=64,
                    start_resblock_num=2,
                    unet_resblock_num=4,
                    end_resblock_num=2
                    ):
        super(EncoderAlphaWeight, self).__init__()
        self.input_num=input_num

        self.ref_feature_extractor=InputFeatureExtractor(ref_channel,base_feature_channels,input_feature_extractor_layers)

        self.input_rgb_feature_extractors=nn.ModuleList()
        for _ in range(input_num):
            self.input_rgb_feature_extractors.append(InputFeatureExtractor(input_rgb_channel,base_feature_channels,input_feature_extractor_layers))

        self.input_alpha_feature_extractors=InputFeatureExtractor(input_alpha_channel*input_num,base_feature_channels,input_feature_extractor_layers)

        self.softmax=nn.Softmax(dim=1)

        self.ref_alpha_merge=nn.Sequential(
                                    nn.Conv2d(base_feature_channels*3,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())

        start_feature_res_blocks=[]
        for _ in range(start_resblock_num):
            start_feature_res_blocks.append(ResBlock(base_feature_channels,2))
        self.start_feature_res_blocks=nn.Sequential(*start_feature_res_blocks)

        self.unet_conv_1=nn.Sequential(
                                        nn.Conv2d(base_feature_channels,base_feature_channels*2, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        self.unet_conv_2=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*4, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*4,base_feature_channels*4, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        unet_feature_res_blocks=[]
        for _ in range(unet_resblock_num):
            unet_feature_res_blocks.append(ResBlock(base_feature_channels*4,2))
        self.unet_feature_res_blocks=nn.Sequential(*unet_feature_res_blocks)

        self.unet_conv_up_1=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*6,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels*2,base_feature_channels*2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        self.unet_conv_up_2=nn.Sequential(
                                        nn.Conv2d(base_feature_channels*3,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(base_feature_channels,base_feature_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())

        end_feature_res_blocks=[]
        for _ in range(end_resblock_num):
            end_feature_res_blocks.append(ResBlock(base_feature_channels,2))
        self.end_feature_res_blocks=nn.Sequential(*end_feature_res_blocks)

        self.output_conv=nn.Sequential(
                                    nn.Conv2d(base_feature_channels,output_channel, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())

    def forward(self, input_list):

        ref_feature=self.ref_feature_extractor(input_list[0])


        rgb_feature_list=[]
        alpha_list=[]
        for input_idx,input_tensor in enumerate(input_list[1]):
            input_rgb_feature=self.input_rgb_feature_extractors[input_idx](input_tensor[:,0:3,:,:])
            rgb_feat = input_rgb_feature.unsqueeze(1)
            cnt_alpha = input_tensor[:,3:4,:,:]
            alpha_list.append(cnt_alpha)
            rgb_feature_list.append(rgb_feat * cnt_alpha.unsqueeze(1))

        alpha = torch.cat(alpha_list,dim=1)
        alpha_feature=self.input_alpha_feature_extractors(alpha)

        #alpha_weight=self.softmax(alpha.unsqueeze(2))

        rgb_feature = torch.cat(rgb_feature_list,dim=1)

        #rgb_feature=rgb_feature * alpha
        rgb_feature=torch.sum(rgb_feature,dim=1)

        feature=torch.cat([ref_feature,rgb_feature,alpha_feature],dim=1)
        feature=self.ref_alpha_merge(feature)

        feature=self.start_feature_res_blocks(feature)

        feature_1=feature
        feature=self.unet_conv_1(feature)

        feature_2=feature
        feature=self.unet_conv_2(feature)

        feature=self.unet_feature_res_blocks(feature)

        feature=F.interpolate(feature,(feature_2.size(2),feature_2.size(3)))
        feature=torch.cat((feature,feature_2),dim=1)

        feature=self.unet_conv_up_1(feature)

        feature=F.interpolate(feature,(feature_1.size(2),feature_1.size(3)))
        feature=torch.cat((feature,feature_1),dim=1)

        feature=self.unet_conv_up_2(feature)

        feature=self.end_feature_res_blocks(feature)

        feature=self.output_conv(feature)

        return feature





class Decoder(nn.Module):
    def __init__(self, output_num=32,
                    output_channel=4,
                    input_channel=3,
                    base_feature_channels=64,
                    start_resblock_num=3,
                    middle_resblock_num=2,
                    end_resblock_num=3
                    ):
        super(Decoder, self).__init__()

        self.input_conv=nn.Conv2d(input_channel,base_feature_channels, kernel_size=3, stride=1, padding=1)

        start_feature_res_blocks=[]
        for _ in range(start_resblock_num):
            start_feature_res_blocks.append(DecoderResBlock(base_feature_channels))
        self.start_feature_res_blocks=nn.Sequential(*start_feature_res_blocks)

        middle_feature_res_blocks=[]
        for _ in range(middle_resblock_num):
            middle_feature_res_blocks.append(DecoderResBlock(base_feature_channels))
        self.middle_feature_res_blocks=nn.Sequential(*middle_feature_res_blocks)

        end_feature_res_blocks=[]
        for _ in range(end_resblock_num):
            end_feature_res_blocks.append(DecoderResBlock(base_feature_channels))
        self.end_feature_res_blocks=nn.Sequential(*end_feature_res_blocks)

        self.output_conv=nn.Sequential(
                                    nn.Conv2d(base_feature_channels,base_feature_channels*4, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(base_feature_channels*4,output_channel*output_num, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())



    def forward(self, x):

        x=self.input_conv(x)

        x_1=x
        x=self.start_feature_res_blocks(x)
        x=x+x_1

        x_2=x
        x=self.middle_feature_res_blocks(x)
        x=x+x_2

        x_3=x
        x=self.end_feature_res_blocks(x)
        x=x+x_3

        x=self.output_conv(x)

        return x




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DecoderResBlock_depthwise(nn.Module):
    def __init__(self,feature_channel=64):
        super(DecoderResBlock_depthwise, self).__init__()
        self.layers=nn.Sequential(InvertedResidual(feature_channel, feature_channel*2, 1), nn.ReLU(),
                                InvertedResidual(feature_channel*2, feature_channel, 1))

    def forward(self,x):
        x_input=x
        x=self.layers(x)
        x=x+x_input
        return x



class DecoderDepthwise(nn.Module):
    def __init__(self, output_num=32,
                    output_channel=4,
                    input_channel=3,
                    base_feature_channels=64,
                    start_resblock_num=3,
                    middle_resblock_num=2,
                    end_resblock_num=3
                    ):
        super(DecoderDepthwise, self).__init__()

        self.input_conv=nn.Conv2d(input_channel,base_feature_channels, kernel_size=3, stride=1, padding=1)

        start_feature_res_blocks=[]
        for _ in range(start_resblock_num):
            start_feature_res_blocks.append(DecoderResBlock_depthwise(base_feature_channels))
        self.start_feature_res_blocks=nn.Sequential(*start_feature_res_blocks)

        middle_feature_res_blocks=[]
        for _ in range(middle_resblock_num):
            middle_feature_res_blocks.append(DecoderResBlock_depthwise(base_feature_channels))
        self.middle_feature_res_blocks=nn.Sequential(*middle_feature_res_blocks)

        end_feature_res_blocks=[]
        for _ in range(end_resblock_num):
            end_feature_res_blocks.append(DecoderResBlock_depthwise(base_feature_channels))
        self.end_feature_res_blocks=nn.Sequential(*end_feature_res_blocks)

        self.output_conv=nn.Sequential(
                                    nn.Conv2d(base_feature_channels,base_feature_channels*4, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(base_feature_channels*4,output_channel*output_num, kernel_size=3, stride=1, padding=1),
                                    nn.Tanh())



    def forward(self, x):

        x=self.input_conv(x)

        x_1=x
        x=self.start_feature_res_blocks(x)
        x=x+x_1

        x_2=x
        x=self.middle_feature_res_blocks(x)
        x=x+x_2

        x_3=x
        x=self.end_feature_res_blocks(x)
        x=x+x_3
        #print("x", x.size())
        x=self.output_conv(x)

        return x











class ResNetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, opt, padding_type='reflect'):
        # ngf number of generator fileters in first conv layer
        '''
        :param input_nc: Input channels for each frames times frame number
        :param output_nc: Output channels for predict next frame
        :param ngf:number of generator filters in first conv layer
        :param n_downsampling:
        :param n_blocks:
        :param padding_type:
        '''
        super(ResNetEncoder, self).__init__()        
        self.n_downsampling = opt.n_downsample_e
        self.opt = opt
        activation = nn.LeakyReLU(True)
        mpi_num = opt.mpi_num
        ngf = opt.nef
        self.use_diff = opt.use_diff
        n_blocks = opt.n_blocks_local
        model_down_input = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.downconv1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), activation)
        self.downconv2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1), activation)
        #self.downconv3 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1), activation)

        self.LeakyRELU = nn.LeakyReLU(0.1)

        mult = 4
        model_res_down = []
        for i in range(n_blocks - n_blocks//2):
            model_res_down += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation)]

        #### Output all at once
        model_res_up = []
        for i in range(n_blocks//2):
            model_res_up += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation)]

        
        #self.upconv1 = nn.Sequential(nn.Conv2d(ngf * 6, ngf * 4, kernel_size=3, stride=1, padding=1), activation)
        self.upconv2 = nn.Sequential(nn.Conv2d(ngf * 6, ngf * 2, kernel_size=3, stride=1, padding=1), activation)
        self.upconv3 = nn.Sequential(nn.Conv2d(ngf * 3, ngf, kernel_size=3, stride=1, padding=1), activation)

        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]

        #model_final_img_1 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, tOut * 4, kernel_size=7, padding=0)]
        #self.model_final_img_1 = nn.Sequential(*model_final_flow_1)

        #model_final_img_2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, tOut * 4, kernel_size=7, padding=0)]
        #self.model_final_img_2 = nn.Sequential(*model_final_img_2)

        #model_final_img_3 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*4, tOut * 4, kernel_size=7, padding=0)]
        #self.model_final_img_3 = nn.Sequential(*model_final_img_3)

        self.model_down_input = nn.Sequential(*model_down_input)
        self.model_res_down = nn.Sequential(*model_res_down)
        self.model_res_up = nn.Sequential(*model_res_up)
        self.dq = DifferentiableQuantize()
        self.model_final_img = nn.Sequential(*model_final_img)
    
    def rescale(self, a):
        return (a + 1.0) / 2.0 * 255.0
    
    def scale(self, a):
        return (a / 255.0) * 2.0 - 1.0

    def forward(self, MPI, Alpha, Ref_Image):
        original_img = Ref_Image
        input = torch.cat([MPI, Alpha, Ref_Image], dim=1)
        down1 = self.model_down_input(input)
        #down1 - 64x256x512
        down2 = self.downconv1(down1)
        #down2 - 128x128x256
        down3 = self.downconv2(down2)
        #down3 - 256x64x128
        #down4 = self.downconv3(down3)
        #down4 - 512x32x64
        feat = self.model_res_up(self.model_res_down(down3))
        #up4 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        #up4_concat = torch.cat([up4, down3], dim=1)
        #up4_conv = self.upconv1(up4_concat)

        #flow3 = self.model_final_flow_3(up4_conv)
        #flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        # 256x128x256
        up3_concat = torch.cat([up3, down2], dim=1)
        up3_conv = self.upconv2(up3_concat)
        # 128x128x256
        up2 = F.interpolate(up3_conv, scale_factor=2, mode='bilinear', align_corners=False)
        # 128x256x512
        up2_concat = torch.cat([up2, down1], dim=1)
        # 128+64 
        up2_conv = self.upconv3(up2_concat)
        #flow1 = self.model_final_flow_1(up2_conv)
        img = self.model_final_img(up2_conv)
        # rescale img to 0 - 255
        img = img + original_img
        if not self.opt.not_quantization:
            img = self.scale(self.dq(self.rescale(img)))
        
        return img
        

class ResNetDecoder(nn.Module):
    # This model inverse the ResNet Encoder
    def __init__(self, input_nc, output_nc, opt, padding_type='reflect'):
        # ngf number of generator fileters in first conv layer
        '''
        :param input_nc: Input channels for each frames times frame number
        :param output_nc: Output channels for predict next frame
        :param ngf:number of generator filters in first conv layer
        :param n_downsampling:
        :param n_blocks:
        :param padding_type:
        '''
        super(ResNetDecoder, self).__init__()        
        n_downsampling = opt.n_downsample_d
        mpi_num = opt.mpi_num
        self.mpi_num = mpi_num
        self.use_diff = opt.use_diff
        ngf = opt.ndf 
        n_blocks = opt.n_blocks_local
        activation = nn.LeakyReLU(True)
        
        model_down_input = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.downconv1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), activation)
        self.downconv2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1), activation)
        #self.downconv3 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1), activation)

        self.LeakyRELU = nn.LeakyReLU(0.1)

        mult = 4
        model_res_down = []
        for i in range(n_blocks - n_blocks//2):
            model_res_down += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation)]

        #### Output all at once
        model_res_up = []
        for i in range(n_blocks//2):
            model_res_up += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation)]

        
        self.upconv2 = nn.Sequential(nn.Conv2d(ngf * 6, ngf * 2, kernel_size=3, stride=1, padding=1), activation)
        self.upconv3 = nn.Sequential(nn.Conv2d(ngf * 3, ngf, kernel_size=3, stride=1, padding=1), activation)
        
        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, mpi_num*3, kernel_size=7, padding=0), nn.Tanh()]
        model_final_mask = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, mpi_num*1, kernel_size=7, padding=0), nn.Sigmoid()]

        self.model_down_input = nn.Sequential(*model_down_input)
        self.model_res_down = nn.Sequential(*model_res_down)
        self.model_res_up = nn.Sequential(*model_res_up)
        self.model_final_img = nn.Sequential(*model_final_img)
        self.model_final_mask = nn.Sequential(*model_final_mask)

    def forward(self, img):
        original_img = img.repeat(1, self.mpi_num, 1, 1)
        down1 = self.model_down_input(img)
        down2 = self.downconv1(down1)
        down3 = self.downconv2(down2)
        #down4 = self.downconv3(down3)
        feat = self.model_res_up(self.model_res_down(down3))
        up3 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        up3_concat = torch.cat([up3, down2], dim=1)
        up3_conv = self.upconv2(up3_concat)
        #flow2 = self.model_final_flow_2(up3_conv)
        #flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = F.interpolate(up3_conv, scale_factor=2, mode='bilinear', align_corners=False)
        up2_concat = torch.cat([up2, down1], dim=1)
        up2_conv = self.upconv3(up2_concat)
        if self.use_diff is True:
            img = self.model_final_img(up2_conv)*2.0
        else:
            img = self.model_final_img(up2_conv)
            img = img + original_img
        mask = self.model_final_mask(up2_conv)
        return img, mask










class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)          


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, mask=None):
        while x.size()[3] > 500:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            _, ch, h, w = x_vgg[i].size()
            if mask is not None:
                curr_mask = F.interpolate(mask, (h, w),mode='nearest')
                curr_mask = curr_mask.expand(-1, ch, -1, -1)
                loss += self.weights[i] * self.criterion(x_vgg[i]*curr_mask, y_vgg[i].detach()*curr_mask)
            else:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def gradient_x(self, img, backmask):
        gx = (img[:,:,:-1,:] - img[:,:,1:,:])*backmask[:,:,1:,:]
        return gx

    def gradient_y(self, img, backmask):
        gy = (img[:,:,:,:-1] - img[:,:,:,1:])*backmask[:,:,:,1:]
        return gy

    def compute_smooth_loss(self, flow_x, img, backmask):
        flow_gradients_x = self.gradient_x(flow_x, backmask)
        flow_gradients_y = self.gradient_y(flow_x, backmask)

        image_gradients_x = self.gradient_x(img, backmask)
        image_gradients_y = self.gradient_y(img, backmask)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, True))

        smoothness_x = flow_gradients_x * weights_x
        smoothness_y = flow_gradients_y * weights_y

        return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img, backmask):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(flow[:,i:i+1,:,:], img, backmask)
        return smoothness/2

    def forward(self, flow, image, mask):
        return self.compute_flow_smooth_loss(flow, image, mask)



from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out




if __name__ == "__main__":
    encoder=EncoderAlphaWeight()
    decoder=Decoder()
    from differentiable_quantize import DifferentiableQuantize
    dq=DifferentiableQuantize()
    mpi_input_list=[]
    for _ in range(32):
        mpi_input_list.append(torch.randn(1,4,256,256))
    input_list=[torch.randn(1,3,256,256),mpi_input_list]
    encoder_out=encoder(input_list)
    encoder_out=encoder_out*127.5+127.5
    encoder_out=dq(encoder_out)
    #print(encoder_out.size())
    encoder_out=(encoder_out-127.5)/127.5
    decoder_out=Decoder(encoder_out)
    #print(decoder_out.size())