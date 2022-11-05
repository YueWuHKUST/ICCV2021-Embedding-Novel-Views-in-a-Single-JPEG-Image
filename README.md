# ICCV2021-Embedding-Novel-Views-in-a-Single-JPEG-Image

This is the official implmentation of ICCV 2021 paper:
Embedding-Novel-Views-in-a-Single-JPEG-Image

# Environment
Pytorch 1.9.0

# Dataset
We randomly select 1500 sequences from RealEstate10K to generate MPIs for testing. 
The IDs of the sequences for training and testing are in train_data_ids.txt and test_data_ids.txt respectively.

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






# To do
- [ ] release training 
- [ ] Write document and instructions
