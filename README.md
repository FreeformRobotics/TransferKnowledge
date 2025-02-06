# Transferring Visual Knowledge: Semi-Supervised Instance Segmentation for Object Navigation Across Varying Height Viewpoints

Qiu Zheng, Junjie Hu, Yuming Liu, Zengfeng Zeng, Fan Wang and Tin Lun Lam

The Chinese University of Hong Kong, Shenzhen, China

Shenzhen Institute of Artificial Intelligence and Robotics for Society (AIRS)

Baidu Inc., Beijing, China



## This repository contains

* The RGB-D dataset with labels collected from Habitat-Matterport 3D household scenes at 10 different height viewpoints.
* The code for per-training, semi-supervised training and evaluate the student model
* The student model transferring from 0.88 m to 0.28 m after semi-supervised training



## File Setup

* Make folders ``./Semantic_Datase/labeled``, ``./Semantic_Datase/unlabeled``, ``./Semantic_Datase/val``

* Download train and val dataset and set the soft link of dataset

  ```sh
  ln -s your_path/train/<source height> ./Semantic_Datase/labeled/
  ln -s your_path/train/<target height> ./Semantic_Datase/unlabeled/
  ln -s your_path/val/<source height> ./Semantic_Datase/val/
  ```

* If using the model after training, download the teacher model and student model into the folder ``./output/teacher`` and ``./output/student``, respectively.

The file structure should look like this:

```sh
.
├── config
│   └── Arguments.py
├── dataloader
│   ├── DataLoder.py
│   ├── DataProcessing.py
│   └── Instance.py
├── dependence
│   ├── detectron2_proto
│   ├── fast_rcnn.py
│   ├── rcnn.py
│   └── roi_heads.py
├── loss
│   ├── ApVal.py
│   ├── PrototypeContrastiveLoss.py
│   └── PrototypeInfoNceLoss.py
├── main.py
├── output
│   ├── student
│   ├── teacher
│   └── tmp
├── pre_train.py
├── projection.py
├── prototype
│   └── PrototypeCompute.py
└── Semantic_Dataset
    ├── labeled
    ├── unlabeled
    └── val
```



## Dataset

The dataset is in Baidu Netdisk

* **Training dataset**

  [train.rar](https://pan.baidu.com/s/1RlW6DsmYpAr1zaQF1ILkFQ) :  extraction code [ yzzg ]

* **Val dataset**

  [val.rar](https://pan.baidu.com/s/1h_5bRa0bURYyPzoS9Owwlw)  :  extraction code [ 66vi ]




## Installing Dependencies

1. Install [pytorch](https://pytorch.org/ )

   The model is trained and tested on pytorch v2.1.0, using conda:

   ```sh
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. Install [detectron2](https://github.com/facebookresearch/detectron2/)

   We build detectron2 from Source:

   ```sh
   cd ./dependence
   git clone https://github.com/facebookresearch/detectron2.git
   ```

   use the files in ``./dependence`` to replace the source code:

   ```sh
   cp ./fast_rcnn.py ./detectron2/detectron2/modeling/roi_heads
   cp ./roi_heads.py ./detectron2/detectron2/modeling/roi_heads
   cp ./rcnn.py ./detectron2/detectron2/modeling/meta_arch
   ```
   build the detectron2:

   ```sh
   python -m pip install -e detectron2
   ```

3. Install other requirements:

   ```sh
   cd your_path/TransferKnowledge
   pip install -r requirements.txt
   ```



## Test setup

1. Downlowd the model after training

   * [teacher model](https://pan.baidu.com/s/15F9ki7x30hT8k7kTYjgFPg ): extraction code [ qt0x ]

     download the teacher model into the folder ``./output/teacher/``

   * [student model](https://pan.baidu.com/s/1F3-6gJK7NVPDyo2MSzORqA ): extraction code [ w6av ]

     download the student model into the folder ``./output/student/``

2. To verify that the student model

   ```sh
   cd loss
   python ApVal.py --val_path ../Semantice_Dataset/val/<source height>
   
   e.g. python ApVal.py --val_path ../Semantice_Dataset/val/0.28
   ```

   

## Training setup

1. Project images from the source height viewpoint to target height viewpoint

   the height is range from **0.28 m** to **1.18 m**, with an interval of 0.1 m.

   ```sh
   python projection.py --source_height <source height> --traget_height <target height>   
   
   e.g. python projection.py --source_height 0.88 --traget_height 0.28
   ```

2. Train the teacher model

   ```sh
   python pre_train.py
   ```

3. Train the student model

   ```sh
   python main.py --labeled_path ./Semantic_Dataset/labeled/<source height>-><target height>_projection --unlabeled_path ./Semantic_Dataset/unlabeled/<target height> --val_path ./Semantic_Dataset/val/<target height>       
   
   e.g. python train.py --labeled_path ./Semantic_Dataset/labeled/0.88->0.28_projection --unlabeled_path ./Semantic_Dataset/unlabeled/0.28 --val_path ./Semantic_Dataset/val/0.28 
   ```

   