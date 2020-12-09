from utils import *
import os
from glob import glob
from tqdm import tqdm
from model.Tfilm import tfilm_net

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "./checkpoint.h5"
in_dir_hr = "/path_to/test_hr/"
in_dir_lr = "/path_to/test_lr/"

lr_folder = None
save_folder = ""

if __name__ == '__main__':

    model = tfilm_net()
    model.load_weights(model_path)
    model.summary()

    # caculate metrics
    snr, lsd = evaluation(model, crop_length=8192, channel=None,
                          in_dir_hr=in_dir_hr, in_dir_lr=in_dir_lr)
    print("SNR: ", snr, " LSD: ", lsd)

    # generate SR audios from LR audios in 'lr_folder'
    if lr_folder is not None:
        paths = glob(lr_folder + "*.wav")
        paths.sort()
        names = os.listdir(lr_folder)
        names.sort()
        num = len(names)
        for i in tqdm(range(num)):
            generate_sr_sample(model, crop_length=8192,
                               in_dir_lr=paths[i], save_path=save_folder + names[i][2:])




