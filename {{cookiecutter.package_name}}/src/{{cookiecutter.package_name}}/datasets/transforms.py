import numpy as np
import torch

from {{cookiecutter.package_name}}.preprocess import aggregate_wt


class ToNumpy:
    """Convert list of tracks to numpy"""

    def __call__(self, song):
        return np.array(song, dtype=float)


class ToTensor:
    """Convert numpy array song to tensor"""

    def __call__(self, song):
        # Might have to do some transposing
        return torch.tensor(song, requires_grad=True, dtype=torch.double)


# --------------------------------------------
# E V E N T  B A S E D  T R A N S F O R M S
class trnsfrm_note_to_idx:
    """Map event based to numbers"""

    def __init__(self, note_to_idx):
        self.note_to_idx = note_to_idx

    def __call__(self, song):
        if type(song[0]) == str:
            return [self.note_to_idx[note] for note in song]
        for i, track in enumerate(song):
            new_track = [k for k in range(len(track))]
            for j, note in enumerate(track):
                new_track[j] = self.note_to_idx[note]
            song[i] = new_track

        return song


class trnsfrm_idx_to_note:
    """Map numbers back to event based"""

    def __init__(self, idx_to_note):
        self.idx_to_note = idx_to_note

    def __call__(self, song):
        for i, track in enumerate(song):
            new_track = [k for k in range(len(track))]
            for j, idx in enumerate(track):
                new_track[j] = self.idx_to_note[idx]
            song[i] = new_track

        return song


class pad_EB:
    """Pad tracks by longest track"""

    def __init__(self, token="<pad>"):
        self.token = token

    def __call__(self, song):
        # Get max length
        max_len = max([len(track) for track in song])
        # Loop through track
        for i, track in enumerate(song):
            track_len = len(track)
            # Do nothing if track is max_len
            if track_len == max_len:
                continue
            # Else create new track of size max_len filled with self.token
            new_track = [self.token for i in range(max_len)]
            # Replace first j elements in new_track with msg from original track
            for j in range(track_len):
                new_track[j] = track[j]
            song[i] = new_track

        return song


class pad_EB_fixed:
    """Pad tracks to a fixed length"""

    def __init__(self, N=50, pad_token="<pad>"):
        self.pad_token = pad_token
        self.N = N

    def __call__(self, song):
        # Loop through track

        new_song = [None for _ in range(len(song))]

        for i, track in enumerate(song):
            # Else create new track of size max_len filled with self.token
            new_track = [self.pad_token for i in range(self.N)]
            # Replace first j elements in new_track with msg from original track
            for j in range(self.N):
                try:
                    new_track[j] = track[j]
                except IndexError:
                    break
            new_song[i] = new_track

        return new_song


class get_single_track:
    """Get single track from song"""

    def __init__(self, trk_idx):
        self.trk_idx = trk_idx

    def __call__(self, song):
        return song[self.trk_idx]


class get_populated_single_track:
    def __init__(
        self,
    ):
        pass

    def __call__(self, song):
        for track in song:
            if len(track) > 3:
                return track
            continue
        return song[1]


class unsqueeze_list:
    """Unsqueeze list"""

    def __call__(self, song):
        return sum(song, [])


class cap_list:
    """Imposes a maximum sequence length on tracks"""

    def __init__(self, cap=200, end_token="<end>"):
        self.cap = cap
        self.end_token = end_token

    def __call__(self, song):
        if type(song[0]) == str:
            return self.cap_a(song)
        else:
            return [self.cap_a(track) for track in song]

    def cap_a(self, array):
        array = array[: self.cap]
        array[-1] = "<end>"
        return array


class split_tracks_src_trg:
    """Excludes a single instrument from a song. Works for instrument specific eb"""

    def __init__(self, einstrument):
        self.einstrument = einstrument

    def __call__(self, song):

        track_X = []
        track_y = ["<begin>"]

        for event in song:

            event_type = event.split("_")[0]

            if event_type == self.einstrument:
                track_y.append(event)

            elif event_type == "wt":
                track_X.append(event)
                track_y.append(event)

            else:
                track_X.append(event)

        # Aggregate wait
        track_X = aggregate_wt(track_X)
        track_y = aggregate_wt(track_y)

        track_y.append("<end>")
        return track_X, track_y


class split_tracks_include_instrument:
    def __init__(self, einstrument):
        self.einstrument = einstrument

    def __call__(self, song):

        track_y = ["<begin>"]

        for event in song:

            event_type = event.split("_")[0]

            if event_type == self.einstrument:
                track_y.append(event)

            elif event_type == "wt":
                track_y.append(event)

        # Aggregate wait
        song = aggregate_wt(song)
        track_y = aggregate_wt(track_y)
        #
        # print()
        # print(song)
        # print()
        # print(track_y)

        track_y.append("<end>")
        return song, track_y


class get_single_track_ispecific:
    def __init__(self, instrument):
        self.instrument = instrument

    def __call__(self, song):

        new_song = ["<begin>"]

        for event in song:

            event_type = event.split("_")[0]

            if event_type == self.instrument:
                new_song.append(event)

            elif event_type == "wt":
                new_song.append(event)

        new_song.append("<end>")
        song = aggregate_wt(new_song)

        return song


class split_tracks_equal_srctrg:
    def __init__(self, instrument):
        self.instrument = instrument

    def __call__(self, song):

        song = get_single_track_ispecific(song)

        return song, song


class remove_tracks:
    def __init__(
        self,
    ):
        pass

    def __call__(self, song):

        new_song = ["<begin>"]

        for event in song:
            event_type = event.split("_")[0]

            # Catch all instruments we don't want
            if event_type in ["Piano", "Guitar", "Bass", "wt"]:
                new_song.append(event)

        song = aggregate_wt(new_song)
        song.append("<end>")
        return song


class remove_wt_tokens:
    def __init__(
        self,
    ):
        pass

    def __call__(self, song):
        new_song = []

        for t in song:
            if t.split("_")[0] == "wt":
                continue
            new_song.append(t)

        return new_song


# --------------------------------------------
# N U M P Y  T R A N S F O R M S
class CropEnding:
    """Crop the ending from the song"""

    def __init__(self, output_ticks):
        assert isinstance(output_ticks, int)
        self.output_ticks = output_ticks

    def __call__(self, song):

        if song.shape[1] < self.output_ticks:

            new_song = np.zeros((9, self.output_ticks, 128))

            return new_song
            # return np.pad(song, (1, self.output_ticks))

        return song[:, : self.output_ticks, :]
