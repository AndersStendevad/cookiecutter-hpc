import torch

from torch.nn.utils.rnn import pad_sequence

from {{cookiecutter.package_name}}.preprocess import (
    trnsfrm_note_to_idx,
    generate_vocab_ispecific,
    generate_vocab_dicts,
)


class Vocab:
    """
    Vocabulary class for the model.
    """

    def __init__(self):
        """
        Initializes the vocabulary.
        """
        self.vocab = generate_vocab_ispecific(exclude=None)
        self.idx_to_note, self.note_to_idx = generate_vocab_dicts(self.vocab)

    def generate_batch(self, data_batch):
        """
        Creates a batch of (batch_size, seq_len)

        Parameters:
            data_batch: list of lists of notes
        """
        batch = []
        trsfrm = trnsfrm_note_to_idx(self.note_to_idx)
        pad_idx = self.note_to_idx["<pad>"]
        for item in data_batch:
            if item == ["<begin>", "<end>"]:
                last = torch.tensor(trsfrm(item))
                continue
            batch.append(torch.tensor(trsfrm(item)))
        if len(batch) == 0:
            return pad_sequence([last])
        batch = pad_sequence(batch, padding_value=pad_idx).transpose(0, 1)
        return batch

    def __len__(self):
        """
        Returns the length of the vocabulary.
        """
        return len(self.vocab)
