# GSMFormer: A Structure-Aware Framework with Agent-Gated Fusion for Building Semantic Segmentation
This paper have been submitted in IEEE Transactions on Geoscience and Remote Sensing (TGRS) in 2026.02. (Major Revision)
Time Line:
```
2026.02.16 Submitted for Round 1 Revision
2026.04.18 Major Revision
2026.05.21 Submitted for Round 2 Revision
2026.06.19 Major Revision
2026.09.09 Submitted for Round 3 Revision
```
<img width="9785" height="8079" alt="structure_1_1" src="https://github.com/user-attachments/assets/3c717d13-4c15-4fbc-abb6-2073535f4a00" />

Paper link: [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.177138973.36328027/v2)

To get the pretrained weights of GSMFormer, please browser: [Hugging Face](https://huggingface.co/buckets/JasonGao726/GSMFormer-bucket) or [Baidu Cloud](https://pan.baidu.com/s/1ZD9kC7ic4L9pSWPLBlsrxg) (Code: md3v)

Links to our preprocessed dataset:

WHU Building Dataset: [Original link](https://gpcv.whu.edu.cn/data/building_dataset.html)

Masachusetts Building Dataset: [HuggingFace](https://huggingface.co/datasets/JasonGao726/Massachusettes_Building_Dataset_Pre-processed/tree/main) or [BaiduCloud](https://pan.baidu.com/s/1CjZhQiJV6bn3GcOiGu2q2A) (Code: bytm)    [Original link](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/data)

ISPRS Potsdam Dataset: [HuggingFace]() or [BaiduCloud](https://pan.baidu.com/s/1LC8aINEhvJFI2CXBT8cvbg) (Code: tppx)

## Comparation Expeirments
### WHU Building Dataset

<p align="center">
  <img width="685" height="461" alt="image" src="https://github.com/user-attachments/assets/819f34b1-0619-41bd-84eb-c6cdd45e763e" />
</p>

### Massacusetts Building Dataset

<p align="center">
  <img width="670" height="464" alt="image" src="https://github.com/user-attachments/assets/0561b4cd-f843-4290-8c89-ed06b1636475" />
</p>

### ISPRS Potsdam Dataset

<p align="center">
  <img width="685" height="461" alt="image" src="https://github.com/user-attachments/assets/819f34b1-0619-41bd-84eb-c6cdd45e763e" />
</p>


## 1. Environment Preparation
Before you start to train or test the model, create a new venv.
```
conda create -n gsmformer python=3.10 -y
conda activate gsmformer
```
Install pytorch
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```
或者国内镜像：
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
Install necessary packages accroding to the requirements.txt. 
```
pip install -r requirements.txt
```

## 2. Training preparation
```

```


## 3. Test preparation

## 4. Getting pretrained weights. 



## 5. Citations
If you hope to cite our works through BibTeX, please copy the following content:
```
@article{gao2026gsmformer,
  title   = {GSMFormer: A Structure-Aware Framework with Agent-Gated Fusion for Building Semantic Segmentation},
  author  = {Gao, Yechuan and Zhou, Zetong and Gao, Qian and Shen, Siyi and Huang, Xiang},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```

If you have any questions, please send emails to erpaogao@gmail.com OR zhouzetong_rs@163.com.
