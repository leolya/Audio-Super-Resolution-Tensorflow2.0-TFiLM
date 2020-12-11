# Audio-Super-Resolution-Tensorflow2.0-TFiLM

This is Tensorflow 2.0 verison of Temporal FiLM for Speech Super Resolution. 

Note that this is unofficial implementation.

## Reference:

https://github.com/kuleshov/audio-super-res

https://arxiv.org/abs/1909.06628

## Requirements

`tensorflow 2.0+`

`pip install numpy h5py tqdm scipy librosa soundfile`

## How to use the code?

Download the preprocessed VCTK Single Speaker dataset from the following link and run `h5_data.py` to generate dataset in 'h5' format. 

https://pan.baidu.com/s/1Q8uPLtaJXZ9Odx17Itawtg  extraction code: 3tl6 

https://drive.google.com/file/d/123NN9H1tx2lNnwl3eikvn0Ay-a52untB/view?usp=sharing

Before running the code, please set paths to the dataset.

```python
# the folder of HR audios and LR audios
in_dir_hr_train = "/path_to/train_hr/"
in_dir_lr_train = "/path_to/train_lr/"
in_dir_hr_test = "/path_to/test_hr/"
in_dir_lr_test = "/path_to/test_lr/"

# the path of output .h5 file
out_dir_train = "./train.h5"
out_dir_test = "./test.h5"
```

Run `train.py`  for training.

Run `test.py`  for evaluation.

## Results

*(Single Speaker ratio=4)*

|            | SNR (dB) | LSD (dB) |
| ---------- | -------- | -------- |
| paper      | 16.8     | 3.5      |
| my results | 17.37    | 3.425    |

