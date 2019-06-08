import pretty_midi
import glob2
import numpy as np
from collections import deque
from tqdm import tqdm_notebook as tqdm
import pandas as pd

#################################################################
##### FUNKCJE POMOCNICZE ########################################
#################################################################
def midi_file_to_pianoroll(file, fps = 1):
    """
    Transform MIDI file to piano roll
    """
    midi = pretty_midi.PrettyMIDI(file)
    piano = midi.instruments[0] # Zakładam że jest tylko jeden instrument (pianino)
    pianoroll = piano.get_piano_roll(fs=fps)
    return pianoroll

def pianoroll_to_dict(pianoroll):
    """
    Transform pianoroll to time dictionary:
    {0: 24,56, 1: 55,43, ...}

    Inputs:
    -------
    piano roll matrix

    Outputs:
    --------
    dict_keys_time - time dictionary from song
    unique_notes - unique notes from song as set
    """

    times = pianoroll.shape[1]
    unique_notes = set()
    dict_keys_time = {}

    for time in range(times):
        # nonzero zwraca tuple w 2 wymiarach, dlatego tak dziwnie to zwracam
        (notes_index,) = (pianoroll[:,time].nonzero())
        if ','.join(notes_index.astype(str)) == '':
            dict_keys_time[time] = '-1'
            unique_notes.add('-1')
        else:
            dict_keys_time[time] = ','.join(notes_index.astype(str))
            unique_notes.add(','.join(notes_index.astype(str)))

    return dict_keys_time, unique_notes

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, _ = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def piano_roll_to_melody(piano_roll):
    """
    Convert piano roll to melody according to dap_xia.pdf
    Could be done without pandas, but it is more elegant that way.
    """
    a,b = np.nonzero(piano_roll)
    melody = pd.DataFrame({"a":a, "b":b})
    melody['a'] = pd.cut(melody['a'], 8, labels=False)
    melody = melody.groupby('b').min()['a']
    melody = melody.fillna(0)
    return melody.values.reshape(-1,)

#################################################################
##### KLASY #####################################################
#################################################################

class MIDI_Dataset:
    """
    Class to find all midi files in folder and transform them to form that is acceptable by LSTM.
    This class is later used by Data_API class.

    Arguments:
    ----------
    - path: path to our MIDI dataset (example format: './data/')
    - fps: notes per second (notes precision)
    """
    def __init__(self, path, fps):
        self.fps = fps
        self.unique_notes = set()
        self.dict_list = []
        self.all_songs_cnt = 0
        self.estimated_tempos = []
        self.bag_of_notes = np.zeros(128,)
        self.melodies = []

        # Find all .midi and .mid files and add their names to list
        files = []
        for file in glob2.glob(path+'**/*.mid'):
            files.append(file)
        for file in glob2.glob(path+'**/*.midi'):
            files.append(file)

        # Read MIDI file and transform it into time dictionary
        for file in tqdm(files, desc = 'MIDI files importing'):
            try:
                midi_file = pretty_midi.PrettyMIDI(file)

                # Add values to describe type of dataset
                # - estimated tempo of all songs in list
                # - bag_of_notes to see probability of playing that note
                self.estimated_tempos.append(midi_file.estimate_tempo())
                piano_roll = midi_file.get_piano_roll(fs=self.fps)
                self.bag_of_notes += np.where(piano_roll!=0, 1, piano_roll).sum(axis=1)
                self.melodies.append(piano_roll_to_melody(piano_roll))
                dict_time, unique_notes = pianoroll_to_dict(piano_roll)

                # Append new notes to self fields
                self.unique_notes = self.unique_notes.union(unique_notes)
                self.dict_list.append(dict_time)
                self.all_songs_cnt += 1
            except:
                print('File: ' + file + ' is not a vaild MIDI file')

        # Get probabilities of playing notes by normalizing them
        self.bag_of_notes = self.bag_of_notes/sum(self.bag_of_notes)

        # Label Encoding for unique_notes. '-1':0 (empty note)
        self.notes_mapping = {note:(i) for i, note in enumerate(np.sort(list(self.unique_notes)))}
        self.inverse_mapping={v:k for k,v in self.notes_mapping.items()}

        # Label Encode all unique notes in dataset
        for i in tqdm(range(self.all_songs_cnt), desc = 'Notes label encoding'):
            for time, notes in self.dict_list[i].items():
                self.dict_list[i][time] = self.notes_mapping[notes]

    def unique_notes_len(self):
        return len(self.unique_notes)

    def number_of_songs(self):
        return self.all_songs_cnt

    def __dict_to_sequence(self, song_nr, sequence_length):
        """
        Arguments:
        -------
        song_nr - song number in dataset

        Returns:
        --------
        """
        time_dict = self.dict_list[song_nr]

        times = list(time_dict.keys())
        start_time, end_time = np.min(times), np.max(times)
        n_samples = end_time - start_time

        initial_values = [0]*(sequence_length-1) + [time_dict[start_time]]
        train_values = np.zeros(shape=(n_samples+1, sequence_length))
        target_values = np.zeros(shape=(n_samples+1))
        train_values_per_step = deque(initial_values)
        for i in range(n_samples):
            train_values[i, :] = list(train_values_per_step)
            current_target = time_dict.get(start_time + i, 0)
            target_values[i] = current_target
            train_values_per_step.popleft()
            train_values_per_step.append(current_target)
        train_values[n_samples, :] = list(train_values_per_step)
        return train_values, target_values

    def sequence_to_pianoroll(self, sequence):
        pianoroll_matrix = np.zeros(shape = (128,len(sequence)))
        sequence_splited = [self.inverse_mapping.get(notes).split(',') for notes in sequence]

        for i,notes in enumerate(sequence_splited):
            notes_len = len(notes)
            if notes_len == 1 and notes[0] == '-1':
                pass
            else:
                for note in notes:
                    pianoroll_matrix[int(note),i] = 1

        return pianoroll_matrix

    def get_batch(self, batch_size, seq_len):
        songs_idx = np.random.randint(0, self.all_songs_cnt, size=batch_size)
        batch_train, batch_target = [], []
        for song in songs_idx:
            train_vals, target_vals = self.__dict_to_sequence(song, seq_len)
            batch_train.append(train_vals)
            batch_target.append(target_vals)
        return np.vstack(batch_train), np.hstack(batch_target)

    def sequence_to_midi(self, sequence, program):
        pianoroll = self.sequence_to_pianoroll(sequence)
        generate_to_midi = piano_roll_to_pretty_midi(pianoroll, program = program, fs=self.fps)
        for note in generate_to_midi.instruments[0].notes:
            note.velocity = 100
        return generate_to_midi

class DataAPI:
    def __init__(self, MIDI_dataset, songs_in_batch, batch_size, sequence_length):
        self.MIDI_dataset = MIDI_dataset

        self.unique_notes_len = self.MIDI_dataset.unique_notes_len()
        self.number_of_songs = self.MIDI_dataset.number_of_songs()

        self.songs_in_batch = songs_in_batch
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self._song_loaded = None
        self._batch_pos = 0

    def get_batch(self):
        if self._song_loaded is None:
            songs_load_sequences, songs_load_target = self.MIDI_dataset.get_batch(self.songs_in_batch, self.sequence_length)
            rand_order = np.random.permutation(np.arange(songs_load_target.shape[0]))
            songs_load_sequences = songs_load_sequences[rand_order, :]
            songs_load_target = songs_load_target[rand_order]

            self._song_loaded = (songs_load_sequences,songs_load_target)
            self._batch_pos = 0

        # Calculate end of batch position (if reaches end of songs_load, return last idx)
        songs_load_sequences, songs_load_target = self._song_loaded
        end_pos = min(self._batch_pos + self.batch_size, songs_load_target.shape[0])
        batch_sequences = songs_load_sequences[self._batch_pos:end_pos, :]
        batch_target =  songs_load_target[self._batch_pos:end_pos]

        if end_pos == songs_load_target.shape[0]:
            self._song_loaded = None
        self._batch_pos = end_pos
        return batch_sequences, batch_target

    def random_start(self):
        first_seq = np.zeros(self.sequence_length)
        first_seq[-1] = np.random.randint(self.unique_notes_len)
        return first_seq

    def sequence_to_midi(self, sequence, program):
        return self.MIDI_dataset.sequence_to_midi(sequence, program)

    def get_estimated_tempos(self):
        return np.mean(self.MIDI_dataset.estimated_tempos)

    def get_bag_of_notes(self):
        return self.MIDI_dataset.bag_of_notes

    def get_melodies(self):
        return self.MIDI_dataset.melodies
