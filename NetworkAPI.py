import numpy as np
import time
import matplotlib.pyplot as plt

import torch as torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm

#to calculate distance between melodies
from scipy.spatial.distance import pdist, squareform
from DataAPI import piano_roll_to_melody

def calculate_bag_of_words(piano_roll):
    notes_quant = np.where(piano_roll!=0, 1, piano_roll).sum(axis=1)
    return notes_quant/sum(notes_quant)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + ' Days '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + ' hours '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + ' minutes '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + ' seconds '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + ' miliseconds '
        i += 1
    if f == '':
        f = '0 miliseconds'
    return f


def levenshtein_distance(mel, melodies):
    longest_melody_shape = max(max([mel.shape[0] for mel in melodies]), mel.shape[0])
    new_list = melodies.copy()
    new_list.append(mel)
    new_melodies = [np.hstack([mel, np.zeros((longest_melody_shape - mel.shape[0]))]) for mel in new_list]
    return squareform(pdist(np.stack(new_melodies),'jaccard'))

class NetworkAPI():
    def __init__(self, model, DataAPI, name_to_save, optimizer):
        self.model=model,
        self.model=self.model[0]
        self.DataAPI = DataAPI

        self.name_to_save=name_to_save
        self.iterations=0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.losses = []
        self.iterations_generated = []
        self.estimated_tempos = []
        self.bag_of_words_diffs = []
        self.mean_levenshtain_distance = []
        self.min_levenshtain_distance = []

        self.criterion = nn.NLLLoss()
        self.optimizer = optimizer

    def train_loop(self, iterations, verbose_every_iteration = 10000, generate_every_iteration = 100, song_len=1000):
        print("====== HYPERPARAMETERS ======")
        print("starting epoch=", self.iterations)
        print("epochs to go=", iterations)
        print("Learning rate=", self.optimizer.param_groups[0]['lr'])
        print("=" * 30)
        elapsed_start_time = time.time()
        best_loss = np.inf
        train_start_time = time.time()

        for iteration in range(iterations):
            if iteration % verbose_every_iteration == 0:
                print(48*'-')

            # Generating new sample every cycle to check simmilarities to dataset
            if iteration % generate_every_iteration == 0:
                self.iterations_generated.append(self.iterations+iteration)
                # Generate sample
                sample_midi = self.generate_sample_midi(song_len = song_len)
                sample_piano_roll = sample_midi.get_piano_roll()
                
                # Calculate difference in bag of words (Euclidian distance)
                sample_bag_of_words = calculate_bag_of_words(sample_piano_roll)
                sample_melody = piano_roll_to_melody(sample_piano_roll)
                dataset_bag_of_words = self.DataAPI.get_bag_of_notes()
                bag_of_words_diff = sum((sample_bag_of_words - dataset_bag_of_words)**2)
                self.bag_of_words_diffs.append(bag_of_words_diff)

                # Calculate differences in melody
                dataset_melodies = self.DataAPI.get_melodies()
                melody_diff = levenshtein_distance(sample_melody, dataset_melodies)
                mean_melody_diff = melody_diff[-1][:-1].mean()
                min_melody_diff = melody_diff[-1][:-1].min()
                self.mean_levenshtain_distance.append(mean_melody_diff)
                self.min_levenshtain_distance.append(min_melody_diff)

                # Calculate difference in tempo (ABS value)
                # self.estimated_tempos.append(sample_midi.estimate_tempo())

            features, targets = self.DataAPI.get_batch()
            features, targets = torch.tensor(data=features, dtype=torch.long).to(self.device), torch.tensor(data=targets, dtype=torch.long).to(self.device)
            self.model.zero_grad()

            target_preds = F.log_softmax(self.model(features), dim=1) # dim=1 is row
            loss = self.criterion(target_preds, targets)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.cpu().item())

            if iteration % verbose_every_iteration == 0:
                print('| Iteration: {:3d} | Time: {:6.2f}s | Loss: {:5.2f} |'
                  .format(self.iterations+1, (time.time() - train_start_time), loss.cpu().item()))
                train_start_time = time.time()

            if loss < best_loss:
                state = {
                            'net': self.model.state_dict(),
                            'iteration': self.iterations,
                            'losses': self.losses,
                            'iterations_generated': self.iterations_generated,
                            'estimated_tempos': self.estimated_tempos,
                            'bag_of_words_diffs' : self.bag_of_words_diffs,
                            'mean_levenshtain_distance': self.mean_levenshtain_distance,
                            'min_levenshtain_distance': self.min_levenshtain_distance
                        }
                torch.save(state, self.name_to_save+'.pth.tar')
                best_val_loss = loss

            self.iterations += 1

        print('| Total time elapsed: {:20}'.format(format_time(time.time() - elapsed_start_time)))

    def load_checkpoint(self):
        checkpoint = torch.load(self.name_to_save+'.pth.tar')
        self.iterations=checkpoint['iteration']
        self.losses=checkpoint['losses']
        self.model.load_state_dict(checkpoint['net'])
        self.iterations_generated=checkpoint['iterations_generated']
        self.estimated_tempos=checkpoint['estimated_tempos']
        self.bag_of_words_diffs=checkpoint['bag_of_words_diffs']
        self.mean_levenshtain_distance=checkpoint['mean_levenshtain_distance']
        self.min_levenshtain_distance=checkpoint['min_levenshtain_distance']

    def plot_errors(self):
        plt.figure(figsize=(10, 4))

        if self.iterations < 10000:
            plt.plot(self.losses)
            plt.title('Loss rate')
            plt.xlabel('Iterations (batches proceesed)')
            plt.ylabel('Loss rate')
            plt.show()
        else:
            plt.title('Loss rate averaged')
            plt.xlabel('Iterations (batches proceesed)')
            plt.ylabel('Loss rate')
            denominator = self.iterations / 1000
            losses_avg = [np.mean(self.losses[100*i:100*(i+1)]) for i in range(int(self.iterations//denominator))]
            plt.plot([i for i in range(int(self.iterations//denominator))   ], losses_avg)
            plt.show()

    def plot_estimated_tempo_diff(self):
        plt.figure(figsize=(10,4))

        plt.axhline(self.DataAPI.get_estimated_tempos(), linestyle = '--', color = 'red', linewidth = 3, label = 'Dataset mean tempo')
        plt.plot(self.iterations_generated, self.estimated_tempos, label = 'Estimated tempo of sample')
        plt.legend(loc='center left')
        plt.title('Estimated tempo of generated sample convergence')
        plt.xlabel('Iterations (batches proceesed)')
        plt.ylabel('Estimated tempo')
        plt.show()

    def plot_bag_of_words_diff(self):
        plt.figure(figsize=(10,4))

        plt.plot(self.iterations_generated, self.bag_of_words_diffs)
        plt.title('Bag of notes difference between generated sample and dataset')
        plt.xlabel('Iterations (batches proceesed)')
        plt.ylabel('Difference (Euclidian distance)')
        plt.show()

    def plot_mean_levenshtain_distance(self):
        plt.figure(figsize=(10,4))

        plt.plot(self.iterations_generated, self.mean_levenshtain_distance)
        plt.title('Mean Levenshtein distance between melody of generated sample and melody of dataset')
        plt.xlabel('Iterations (batches proceesed)')
        plt.ylabel('Levenshtein distance')
        plt.show()

    def plot_min_levenshtain_distance(self):
        plt.figure(figsize=(10,4))

        plt.plot(self.iterations_generated, self.min_levenshtain_distance)
        plt.title('Min Levenshtein distance between melody of generated sample and melody of dataset')
        plt.xlabel('Iterations (batches proceesed)')
        plt.ylabel('Levenshtein distance')
        plt.show()

    def generate_sequence(self, song_len, temperature = 1.0):
        sequence = self.DataAPI.random_start().tolist()
        gen_song = [int(sequence[-1])]
        for i in range(song_len - 1):
            output = self.model(torch.tensor(data=np.expand_dims(np.array(sequence), axis=0), dtype=torch.long).to(self.device))
            output = F.softmax(output / temperature, dim=1)
            new_note = torch.multinomial(output, 1)[:, 0]
            new_note = new_note.cpu().item()
            sequence.pop(0)
            sequence.append(new_note)
            gen_song.append(int(new_note))
        return gen_song

    def generate_sample_midi(self, song_len, temperature = 1.0, program = 'Acoustic Grand Piano'):
        song_len = song_len * self.DataAPI.MIDI_dataset.fps
        sample_sequence = self.generate_sequence(song_len, temperature = 1.0)
        return self.DataAPI.sequence_to_midi(sample_sequence, program = program)
