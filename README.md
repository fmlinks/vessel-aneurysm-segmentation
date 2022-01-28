# vessel-aneurysm-segmentation
Cerebrovascular and Aneurysm Segmentation in 3DRA images via a Deep Multi-Task Network

Weight（fmnet5.hdf5）can be download here: https://drive.google.com/file/d/1XZZY_H-Nt6mOZ3E9aFDAVvmYezxWJROi/view?usp=sharing


Folder structure:

vessel-aneurysm-segmentation/

    ├── data
    │   └── step0 (**put the data you want to do inference here**)
    │   │   └── ANSYS_UNIGE_09_image.nii.gz
    │   │   ├── ANSYS_UNIGE_28_image.nii.gz
    │   │   └── ...
    │   ├── step1 (pre-processed data, generate automatically)
    │   │   └── ANSYS_UNIGE_09_image.nii.gz
    │   │   ├── ANSYS_UNIGE_28_image.nii.gz
    │   │   └── ...
    ├── inference

    │   └── inference.ipynb (**use this to do inference**)

    │   ├── inference.py

    ├── results

    │   └── aneurysm (aneurysm prediction)

    │   │   └── ANSYS_UNIGE_09_image-[360, 633].nii.gz

    │   │   ├── ANSYS_UNIGE_28_image-[1540].nii.gz

    │   │   └── ...

    │   ├── vessel (vessel prediction)

    │   │   └── ANSYS_UNIGE_09_image_vessel_59969.nii.gz

    │   │   ├── ANSYS_UNIGE_28_image_vessel_68437.nii.gz

    │   │   └── ...

    ├── weights

    │   └── fmnet5.hdf5

    └── requirements.txt



