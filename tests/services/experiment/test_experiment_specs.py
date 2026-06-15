from core.services.experiment import build_run_specs


def test_smoke_alias_builds_preliminary_spec():
    specs = build_run_specs(
        "smoke",
        dataset_names=["ag_news"],
        seed_limit=1,
        include_baselines=False,
    )

    assert len(specs) == 1
    assert specs[0].preset == "preliminary"
    assert specs[0].dataset.num_documents == 1000
    assert specs[0].models == ("semgraph",)


def test_main_arxiv_uses_large_document_budget():
    specs = build_run_specs(
        "main",
        dataset_names=["arxiv"],
        seed_limit=1,
        include_baselines=False,
    )

    assert len(specs) == 1
    assert specs[0].dataset.num_documents == 100000
    assert specs[0].dataset.min_clusters == 5
    assert specs[0].dataset.max_clusters == 50


def test_mask_rate_ablation_varies_only_mask_rate():
    specs = build_run_specs(
        "ablation",
        dataset_names=["ag_news"],
        axes=["mask_rate"],
        seed_limit=1,
        include_baselines=False,
    )

    assert [spec.params.mask_rate for spec in specs] == [0.1, 0.3, 0.5, 0.75]
    assert {spec.params.top_n_words for spec in specs} == {1000}
    assert {spec.params.graphmae_epochs for spec in specs} == {1000}
    assert {spec.ablation_axis for spec in specs} == {"mask_rate"}
