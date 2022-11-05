# ICCV2021-Embedding-Novel-Views-in-a-Single-JPEG-Image

This is the official implmentation of ICCV 2021 paper:
Embedding-Novel-Views-in-a-Single-JPEG-Image

# Environment
please refer to the enviroment.yml

# Dataset
We provide the test set we use in the [Google drive](https://drive.google.com/drive/folders/1kK8bPSjJEPIGDBjD9sPtVstLRCx0i7Vn?usp=share_link). 
We randomly select 1500 sequences from RealEstate10K to generate MPIs for testing. 
The IDs of the sequences for training and testing are in train_data_ids.txt and test_data_ids.txt respectively. 

And the stereo_mpi represents the MPIs generated using [Stereo Magnification](https://github.com/google/stereo-magnification#stereo-magnification-learning-view-synthesis-using-multiplane-images).
And the pb_mpi_32 represents the MPIs generated using the [Pushing the Boundaries of View Extrapolation with Multiplane Images](https://github.com/google-research/google-research/blob/master/mpi_extrapolation/README.md). The original output of this method generate 128-layer MPIs, we merge the 128-layer MPIs to 32-layer due to network capacity limitation.

We use their public code for the MPIs generations. For the details of how the MPIs are generated, please refer to their public repo.


# The Folder structure
```
ICCV2021-Embedding-Novel-Views-in-a-Single-JPEG-Image/
│
└─── datasets/
    |
    └─── stereo_mpi/
    │
        └─── test_final/   # the test dataset of stereo-magnification
	|
	└─── pb_mpi_32/  # the test dataset of pb_mpi, we merge the original 128 mpi to 32 layers
	    |
	    └─── test_final /
└─── checkpoints/
    |
    └─── stereo_final
│
        └─── models
│
            └─── decoder_003_00015000.pth
│
            └─── discriminator_003_00015000.pth
│
            └─── encoder_003_00015000.pth
    ....
    test.py
    test.sh
    ....
```

# The pretrained models
The pretrained checkpoints are shared in the [Google drive link](https://drive.google.com/drive/folders/1TwKgB2g2H92u_xuUm5A8jmVXYES2Yq5o?usp=sharing) as follows:

|  Methods   | Links  |
|  ----  | ----  |
| Stereo magnification | [Google drive](https://drive.google.com/drive/folders/1x9_QektXnHniVuIpidH4opM6AOQrDgQE?usp=sharing) |
| PB-MPI | [Google drive](https://drive.google.com/drive/folders/111yGAdFgbQ_MI3FRjAoG3pPbPrRFtr6Q?usp=share_link) |
| LLFF | [Google drive](https://drive.google.com/drive/folders/1m39PW8FgdockDU3-UC3OJD30CucGmOcG?usp=share_link) |

# How to test
After put the test data and pretrained checkpoints in the corresponding folder, 

Step1:
```
./test.sh
```
Run the test.sh to generate the embedding image and predicted MPIs. 

The results will be stored in './checkpoints/[checkpoint_name]/test_result/

Step2:
```
./render.sh
```
Run the render.sh to render novel views using predicted MPIs.
The results will be stored in './checkpoints/[checkpoint_name]/render/



# To do
- [ ] release training 
- [ ] Write document and instructions
