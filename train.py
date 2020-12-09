import tensorflow as tf
from dataset import get_audio_dataset
from model.Tfilm import tfilm_net
import os
import h5py
from utils import evaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# training parameters
train_dataset_path = "./train.h5"
test_dataset_path = "./test.h5"
save_path = './'
batch_size = 32
EPOCHS = 50
lr = 3e-4

# for evaluation
in_dir_hr_test = "/path_to/test_hr/"
in_dir_lr_test = "/path_to/test_lr/"

if __name__ == '__main__':

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hf = h5py.File(train_dataset_path, 'r')
    length = hf['data'].shape[0]
    hf.close()

    train_ds = get_audio_dataset(train_dataset_path, batch_size=batch_size, length=length)
    test_ds = get_audio_dataset(test_dataset_path, batch_size=batch_size, length=None)

    model = tfilm_net()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss = tf.keras.metrics.Sum(name='train_loss')
    test_loss = tf.keras.metrics.Sum(name='test_loss')


    @tf.function
    def train_step(inpt, tagt):
        with tf.GradientTape() as tape:
            pred = model(inpt)
            loss = loss_object(tagt, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step(inpt, tagt):
        pred = model(inpt)
        t_loss = loss_object(tagt, pred)
        test_loss(t_loss)

    # loss_min = 1e8
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        test_loss.reset_states()

        for inpt, tagt in train_ds:
            train_step(inpt, tagt)

        for inpt, tagt in test_ds:
            test_step(inpt, tagt)

        # if test_loss.result() < loss_min:
        #     loss_min = test_loss.result()
        #     model.save_weights(save_path + 'best_model.h5')

        model.save_weights(save_path + 'checkpoint.h5')
        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              test_loss.result()))

    # final evaluation
    model.load_weights(save_path + 'checkpoint.h5')
    snr, lsd = evaluation(model, crop_length=8192, channel=None,
                          in_dir_hr=in_dir_hr_test, in_dir_lr=in_dir_lr_test)
    print("Final Results -- SNR: ", snr, " LSD: ", lsd)

