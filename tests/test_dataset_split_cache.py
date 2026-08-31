from src.data.dataset import Sample, split_samples


def _samples(n, generator="cifake"):
    return [Sample(path=f"/data/img_{i}.jpg", label=i % 2, generator=generator) for i in range(n)]


def test_split_samples_reproducible_despite_reordered_input(tmp_path):
    """The bug this caching fixes: scan_dataset()'s filesystem enumeration
    order isn't guaranteed stable, so a fixed seed shuffling a
    differently-ordered input previously gave a different split. With
    caching, the SAME underlying files (regardless of the order they're
    passed in) must produce the SAME split.
    """
    cache_path = tmp_path / "split_cache.json"
    samples = _samples(20)

    first = split_samples(
        samples, holdout_generator="none", train_split=0.6, val_split=0.2, seed=42, cache_path=cache_path
    )
    assert cache_path.exists()

    reordered = list(reversed(samples))  # simulates a different filesystem scan order
    second = split_samples(
        reordered, holdout_generator="none", train_split=0.6, val_split=0.2, seed=42, cache_path=cache_path
    )

    assert [s.path for s in first["train"]] == [s.path for s in second["train"]]
    assert [s.path for s in first["val"]] == [s.path for s in second["val"]]
    assert [s.path for s in first["test"]] == [s.path for s in second["test"]]


def test_split_samples_cache_invalidated_when_files_change(tmp_path):
    cache_path = tmp_path / "split_cache.json"
    samples = _samples(20)
    split_samples(samples, holdout_generator="none", train_split=0.6, val_split=0.2, seed=42, cache_path=cache_path)

    grown = samples + [Sample(path="/data/img_new.jpg", label=0, generator="cifake")]
    second = split_samples(
        grown, holdout_generator="none", train_split=0.6, val_split=0.2, seed=42, cache_path=cache_path
    )

    all_paths = {s.path for split in second.values() for s in split}
    assert all_paths == {s.path for s in grown}
    assert "/data/img_new.jpg" in all_paths


def test_split_samples_holdout_generator_excluded_from_trainable_pool(tmp_path):
    cache_path = tmp_path / "split_cache.json"
    trainable = _samples(16, generator="cifake")
    holdout = [Sample(path=f"/data/holdout_{i}.jpg", label=i % 2, generator="wildfake_gan") for i in range(4)]

    result = split_samples(
        trainable + holdout, holdout_generator="wildfake_gan",
        train_split=0.5, val_split=0.25, seed=42, cache_path=cache_path,
    )

    assert len(result["unseen_generator"]) == 4
    assert all(s.generator == "wildfake_gan" for s in result["unseen_generator"])
    assert all(s.generator != "wildfake_gan" for s in result["train"] + result["val"] + result["test"])


def test_split_samples_no_cache_file_written_when_cache_path_is_none(tmp_path):
    samples = _samples(10)
    split_samples(samples, holdout_generator="none", train_split=0.6, val_split=0.2, seed=42, cache_path=None)
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere
