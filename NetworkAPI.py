import numpy as np
import time
import matplotlib.pyplot as plt

import torch as torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm


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

        self.criterion = nn.NLLLoss()
        self.optimizer = optimizer

    def train(self, iterations):
        """
        Wersja backupowa - do not use
        """
        for iteration in tqdm(range(iterations)):
            features, targets = self.DataAPI.get_batch()
            features, targets = torch.tensor(data=features, dtype=torch.long).to(self.device), torch.tensor(data=targets, dtype=torch.long).to(self.device)
            self.model.zero_grad()
            
            target_preds = F.log_softmax(self.model(features), dim=1) # dim=1 is row
            loss = self.criterion(target_preds, targets)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.cpu().item())
        
        return self.losses


    def train_loop(self, iterations, verbose_every_iteration = 10000):
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
                  .format(iteration+1, (time.time() - train_start_time), loss.cpu().item()))
                train_start_time = time.time()

            if loss < best_loss:
                state = {
                            'net': self.model.state_dict(),
                            'iteration': iteration,
                            'losses': self.losses,
                        }
                torch.save(state, self.name_to_save+'.pth.tar')
                best_val_loss = loss

        print('| Total time elapsed: {:20}'.format(format_time(time.time() - elapsed_start_time)))

    def load_checkpoint(self):
        checkpoint = torch.load(self.name_to_save+'.pth.tar')
        self.iteration=checkpoint['iteration']
        self.losses=checkpoint['losses']
        self.model.load_state_dict(checkpoint['net'])

    def plot_errors(self):
        plt.figure(figsize=(10, 4))

        if self.iterations < 10000:
            plt.plot(self.losses)
            plt.title('Loss rate')
            plt.xlabel('Iterations (batches proceesed)')
            plt.ylabel('Loss rate')
            plt.show()
        else:
            plt.plot(self.losses)
            plt.title('Loss rate averaged')
            plt.xlabel('Iterations (batches proceesed)')
            plt.ylabel('Loss rate')
            plt.show()
            denominator = self.iterations / 1000
            losses_avg = [np.mean(self.losses[100*i:100*(i+1)]) for i in range(self.iterations//denominator)]
            plt.plot([i for i in range(self.iterations//denominator)], losses_avg)
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