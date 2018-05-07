import os
import sys
import time
import numpy as np
from keras.optimizers import SGD
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution1D, AtrousConvolution1D, Flatten, Dense, Dropout ,Input, Lambda, merge


def get_discriminative_model():
    model = Sequential()
    model.add(Dense(4000, activation='relu', kernel_initializer='ones', bias_initializer='zeros' ,input_dim=4000))
    model.add(Dense(800, activation='relu' , kernel_initializer='ones', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid' , kernel_initializer='ones', bias_initializer='zeros'))
    return model


def get_generative_model():
    model = Sequential()
    model.add(Dense(4000, activation='relu', input_dim=4000))

    model.add(Dense(4000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4000, activation='tanh'))
    return model


def get_generator_containing_disciminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def read_files_from_path(dir):
    mas = []
    for root, dirs, files in os.walk(os.path.abspath(dir)):
        for file in files:
            if file.startswith('arctic'):
                mas.append(os.path.join(root, file))
    return np.array(sorted(mas))


def get_audio(filename):
    sr, audio = read(filename)
    audio = np.pad(audio, (0, 100000 - len(audio)), 'constant')
    audio = audio.astype(np.float32)
    max = np.max(audio)
    audio = audio/max
    return sr, audio


def get_training_data(data,frame_size, frame_shift):
    sr, audio = get_audio(data)
    X_train = []
    base = 0
    while base + frame_size <= 100000:
        frame = audio[base:base+frame_size]
        X_train.append(frame)
        base += frame_shift
    X_train = np.array(X_train)
    return sr, np.array(X_train)

if __name__ == '__main__':
    n_epochs = 5
    generator_epochs = 3
    batch_size = 120
    frame_shift = 100
    frame_size = 4000
    n_audios_to_dump = 10
    model_dumping_freq = 5
    old_mean = 1

    generator = get_generative_model()
    discriminator = get_discriminative_model()
    generator_containing_disciminator = get_generator_containing_disciminator(generator, discriminator)
    generator.compile(loss='mae', optimizer='SGD')
    generator_containing_disciminator.compile(loss='binary_crossentropy', optimizer='SGD')
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer='SGD')

    male_wavs = read_files_from_path("audio_samples/wav_male")
    female_wavs = read_files_from_path("audio_samples/wav_female")
    train_idx = int(((len(male_wavs) - (len(male_wavs) % 10)) * 0.8))
    test_idx = int(((len(male_wavs) - (len(male_wavs) % 10)) * 0.2)) + train_idx
    train_male , train_female = male_wavs[0 : train_idx] , female_wavs[0 : train_idx]
    valid_male , valid_female = male_wavs[train_idx : test_idx] , female_wavs[train_idx : test_idx]
    test_arr = male_wavs[test_idx:]

    for i in range(n_epochs):
        print 'Epoch:', i+1

        for num in range(len(train_male)):

            sr_X, X_train = get_training_data(train_male[num],frame_size, frame_shift)
            sr_y, y_train = get_training_data(train_female[num],frame_size, frame_shift)

            n_minibatches = int(X_train.shape[0]/batch_size)
            for index in range(n_minibatches):

                audio_batch_X = np.array(X_train[index*batch_size:(index+1)*batch_size])
                audio_batch_real = np.array(y_train[index*batch_size:(index+1)*batch_size])

                # train generator
                for i in np.arange(batch_size):
                    generator.fit(audio_batch_X[i].reshape(1,frame_size), audio_batch_real[i].reshape(1,frame_size), epochs = 1, verbose = 0)

                # generate audio
                generated_audio_X = []
                for el_x in audio_batch_X:
                    generated_audio = generator.predict(el_x.reshape(1,frame_size))
                    generated_audio_X.append(generated_audio)
                generated_audio_X = np.array(generated_audio_X).reshape(batch_size,frame_size)


                X = np.concatenate((audio_batch_real, generated_audio_X), axis=0)
                y = [1] * batch_size + [0] * batch_size
                y = np.array(y)
                y = y.reshape(batch_size*2 , 1)
                #train discrimanator
                for i in np.arange(batch_size*2):
                    discriminator.fit(X[i].reshape(1,frame_size), y[i], epochs = 1, verbose = 0)


                discriminator.trainable = False
                #trane generator using discriminator
                for i in np.arange(batch_size):
                    generator_containing_disciminator.fit(generated_audio_X[i].reshape(1,frame_size), (np.array([1]*batch_size)).reshape(batch_size , 1)[i], epochs = 1, verbose = 0)
                discriminator.trainable = True
                sys.stdout.write(' + minibatch: ' + str(index+1) + '/' + str(n_minibatches) + '\r')
                sys.stdout.flush()

            print(str(num+1) + " audio processed")

            if (num + 1) % 4 == 0:
                sr_X_test, X_test = get_training_data(valid_male[num/4],frame_size, frame_size)
                sr_y_test, y_test = get_training_data(valid_female[num/4],frame_size, frame_size)

                # generate audio
                test_generated_audio_X = []
                for el_x in X_test:
                    generated_audio = generator.predict(el_x.reshape(1,frame_size))
                    test_generated_audio_X.append(generated_audio)
                test_generated_audio_X = np.array(test_generated_audio_X).reshape(100000/frame_size,frame_size)

                g_losses = []
                for i in np.arange(100000 // frame_size):
                    g_loss = generator.evaluate(test_generated_audio_X[i].reshape(1,frame_size), y_test[i].reshape(1,frame_size), verbose=0)
                    if g_loss > 0:
                        g_losses.append(g_loss)

                d_losses = []
                X = np.concatenate((y_test, test_generated_audio_X), axis=0)
                y = [1] * (100000/frame_size) + [0] * (100000/frame_size)
                y = np.array(y)
                y = y.reshape((100000/frame_size)*2 , 1)
                #train discrimanator
                for i in np.arange((100000/frame_size)*2):
                    d_loss = discriminator.evaluate(X[i].reshape(1,frame_size), y[i], verbose = 0)
                    d_losses.append(d_loss)

                mean_gloss = np.mean(g_losses)
                mean_dloss = np.mean(d_losses)
                print "-" * 80
                print ' + generator loss: ', mean_gloss
                print ' + discriminator loss: ', mean_dloss

                if mean_gloss < old_mean:
                    old_mean = mean_gloss
                    print ' + saving model'
                    str_timestamp = str(int(time.time()))
                    gen_model_savepath = os.path.join('saved_models', str_timestamp + 'saved_model_of_generator.h5')
                    generator.save(gen_model_savepath)
                    print ' + generating audio sample'
                    sr_X, X = get_training_data(test_arr[0],frame_size, frame_size)
                    generated_audio_X = []
                    for el_x in X:
                        generated_audio = generator.predict(el_x.reshape(1,frame_size))
                        generated_audio_X.append(generated_audio)
                    generated_audio_X = np.array(generated_audio_X).reshape(100000,1)
                    name_of_exmple = os.path.join('generated_audios', str_timestamp + 'example.wav')
                    write(name_of_exmple, sr_X, generated_audio_X)
