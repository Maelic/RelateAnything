# Research probes

The analyses behind the paper's claims that are not part of the evaluation
protocol. Each script produces one figure, table or number of the report;
nothing here is needed to train, evaluate or deploy the model. They are kept
as they were run, with paths under `$RA_RUNS`; expect to adapt arguments.

| script | paper claim it produces |
|---|---|
| `bias_baselines.py` | the frequency baseline over object-category pairs that uses no pixels (priors table) |
| `probe_attribution.py`, `report_attribution.py` | input attribution, lesion ladder over components |
| `probe_counterfactual_grounding.py`, `probe_context_window.py` | counterfactual grounding and context-window figure |
| `analyze_backbone_shift.py`, `analyze_patch_similarity.py` | what relation supervision changes in the backbone (patch similarity, representation table) |
| `probe_feature_relatedness.py` | linear probe for relatedness on frozen features |
| `visualize_affordance_similarity.py` | affordance-similarity figure |
| `visualize_deform_points.py` | deformable-read statistics and qualitative offset fields |
| `dump_pair_scores.py`, `e3_confusion.py`, `e3_panels.py`, `e3_type_consistency.py` | pair-score dumps behind the confusion matrices, qualitative panels and type-consistency numbers |
| `e0_report.py`, `probe_wearing_conditional.py`, `probe_wearing_prior.py`, `probe_wearing_model.py` | the `wearing` case study: what the model conditions on versus the prior |
| `swap_probe.py` | subject/object swap accuracy of the objective |
| `probe_sampler_recall.py` | pair-sampler retention |
| `eval_relatedness.py` | relatedness-head analysis |
| `diag_text_spaces.py`, `diag_text_full_vocab.py` | text-space diagnostics (antonym cosine, isotropy) |
| `audit_param_health.py` | untrained-head parameter check |
| `wise_ft_blend.py` | WiSE-FT interpolation ablation |

Model loading in every probe goes through `relsgg.checkpoint.build_model_from_ckpt`,
the same loader the benchmark uses, so a probe can be pointed at any released
`model.pth`.
