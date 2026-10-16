"""tests/test_transformer_dataset.py

Synthetic-data tests for src/fcac/transformer/dataset.py (CLAUDE.md §8 item 9).
"""

import torch

from fcac.transformer.dataset import RecordingDataset, collate, pad_batch


def test_pad_batch_shapes_and_mask():
    sequences = [torch.randn(5, 4), torch.randn(8, 4), torch.randn(3, 4)]
    padded, pad_mask = pad_batch(sequences)
    assert padded.shape == (3, 8, 4)
    assert pad_mask.shape == (3, 8)
    assert not pad_mask[1].any()  # the longest sequence has no padding
    assert pad_mask[0, 5:].all() and not pad_mask[0, :5].any()
    assert pad_mask[2, 3:].all() and not pad_mask[2, :3].any()


def test_recording_dataset_crops_long_recordings():
    z_by_recording = [torch.randn(20, 4), torch.randn(3, 4)]
    targets = torch.tensor([280.0, 300.0])
    dataset = RecordingDataset(z_by_recording, targets, crop_len=5, repeats_per_epoch=2)
    assert len(dataset) == 4  # 2 recordings * repeats_per_epoch=2

    z, y = dataset[0]
    assert z.shape == (5, 4)  # cropped down from 20
    assert y == 280.0

    z_short, y_short = dataset[1]
    assert z_short.shape == (3, 4)  # shorter than crop_len -> kept whole
    assert y_short == 300.0


def test_collate_produces_padded_batch_and_mask():
    z_by_recording = [torch.randn(6, 4), torch.randn(2, 4)]
    targets = torch.tensor([280.0, 290.0])
    dataset = RecordingDataset(z_by_recording, targets, crop_len=100, repeats_per_epoch=1)
    batch = [dataset[i] for i in range(len(dataset))]

    z_padded, pad_mask, y = collate(batch)
    assert z_padded.shape == (2, 6, 4)
    assert pad_mask.shape == (2, 6)
    assert torch.equal(y, targets)
