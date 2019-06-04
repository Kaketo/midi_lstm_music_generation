import copy
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

class MidiAPI:
    """
    Jeszcze w produkcji.
    """
    def __init__(self, midi):
        self.midi = copy.copy(midi)
        self.sample_tempo = midi.estimate_tempo()
        self.sample_tempo_end_time = midi.get_end_time()
        self.step = self.sample_tempo_end_time / self.sample_tempo * 3
        self.hat_times = None
        self.bass_times = None
        self.piano_roll = midi.get_piano_roll()

    def plot_piano_roll(self):
        plt.figure(figsize=(15, 7))
        for i in range(self.piano_roll.shape[1]):
            notes = np.nonzero(self.piano_roll[:, i])
            for note in notes:
                try:
                    plt.plot([i, i+1], [note, note], color='royalblue', linewidth=8)
                except:
                    pass
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        plt.xlabel('Seconds', fontsize=20)
        plt.ylabel('Note', fontsize=20)
        plt.show()

    def compare_piano_rolls(self, midi_2, fps):
        # Plot first piano roll
        plt.figure(figsize=(15, 7))
        for i in range(self.piano_roll.shape[1]):
            notes = np.nonzero(self.piano_roll[:, i])
            for note in notes:
                try:
                    plt.plot([i, i+1], [note, note], color='royalblue', linewidth=8)
                except:
                    pass

        # Plot second piano roll
        pr2 = midi_2.get_piano_roll(fs=fps)
        for i in range(pr2.shape[1]):
            notes = np.nonzero(pr2[:, i])
            for note in notes:
                try:
                    plt.plot([i, i+1], [note, note], color='red', linewidth=8)
                except:
                    pass

        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        plt.xlabel('Seconds', fontsize=20)
        plt.ylabel('Note', fontsize=20)
        plt.show()

    def __generate_hat_times(self):
        self.hat_times = np.arange(0, self.sample_tempo_end_time, self.step)

    def __generate_bass_times(self):
        self.bass_times = self.hat_times + self.step/2

    def add_bass(self):
        bass_instrument = pretty_midi.Instrument(program=1, is_drum=True, name='Bass')
        # Add bass to song
        bass_note = pretty_midi.drum_name_to_note_number('Bass Drum 1')
        for hat in range(0,(len(self.bass_times) - 1)):
            note = pretty_midi.Note(velocity=100, pitch=bass_note, start=self.bass_times[hat], end=self.bass_times[hat+1])
            bass_instrument.notes.append(note)

        self.midi.instruments.append(bass_instrument)
    
    def add_hat(self):
        hat_instrument = pretty_midi.Instrument(program=1, is_drum=True, name='Hat')
        # Add hat to song
        hat_note = pretty_midi.drum_name_to_note_number('Chinese Cymbal')
        for hat in range(0,(len(self.hat_times) - 1)):
            note = pretty_midi.Note(velocity=100, pitch=hat_note, start=self.hat_times[hat], end=self.hat_times[hat+1])
            hat_instrument.notes.append(note)

        self.midi.instruments.append(hat_instrument)  

    def extract_midi(self):
        return self.midi