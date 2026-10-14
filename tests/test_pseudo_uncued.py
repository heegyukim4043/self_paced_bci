import numpy as np

from uncued_mi.pseudo_uncued import build_pseudo_uncued_raw_stream, fixed_budget_exponential_chunks


def test_fixed_budget_rest_spans_reconstruct_chunks():
    blocks = [np.arange(80).reshape(40, 2), np.arange(100, 180).reshape(40, 2)]
    chunks, sources, spans = fixed_budget_exponential_chunks(
        blocks, chunks_per_block=(2, 2), sfreq=1, minimum_seconds=5,
        seed=17, return_spans=True,
    )
    old_chunks, old_sources = fixed_budget_exponential_chunks(
        blocks, chunks_per_block=(2, 2), sfreq=1, minimum_seconds=5, seed=17,
    )
    assert np.array_equal(sources, old_sources)
    for chunk, old, source, (start, stop) in zip(chunks, old_chunks, sources, spans):
        assert np.array_equal(chunk, old)
        assert np.array_equal(chunk, blocks[source][start:stop])
    for source in range(len(blocks)):
        assert sorted(map(tuple, spans[sources == source]))[0][0] == 0
        assert sorted(map(tuple, spans[sources == source]))[-1][1] == len(blocks[source])


def test_random_rest_trial_annotation_is_exact():
    sfreq = 10.0
    signal = np.arange(3000 * 2, dtype=float).reshape(3000, 2)
    rests = [np.zeros((200, 2)), np.ones((200, 2))]
    stream = build_pseudo_uncued_raw_stream(
        continuous_trial_signal=signal,
        cue_samples=np.array([1000, 2000]), labels=np.array([1, 2]),
        rest_blocks=rests, sfreq=sfreq, trial_indices=np.array([0, 1]),
        minimum_rest_seconds=1, maximum_rest_seconds=4, seed=7,
    )
    assert stream.events.shape == (2, 8)
    for row in stream.events:
        _, _, rest_start, rest_stop, trial_start, cue, mi_start, mi_stop = row
        assert rest_stop == trial_start
        assert cue - trial_start == 30
        assert mi_start - cue == 5
        assert mi_stop - cue == 35
        assert np.all(stream.state[rest_start:rest_stop] == 0)
        assert np.all(stream.state[trial_start:mi_start] == 1)
        assert np.all(stream.state[mi_start:mi_stop] == 2)
