# MI308X tuning tables

These CSVs are measured AITER 0.1.19 A8W8 blockscale configs for the single-GPU
DeepSeek-V4-Flash-0731 production profile on **gfx942 / 80 CU (MI308X)**.

- `dsv4-mi308x-80cu-a8w8-blockscale-bpreshuffle.csv`: 13 promoted rows.
- `dsv4-mi308x-80cu-a8w8-blockscale.csv`: 24 promoted rows.

Why they exist: the upstream MI300X tables use `cu_num=304`. AITER keys config
lookup by `(gfx, cu_num, M, N, K)`, so an MI308X reporting 80 CUs does **not** hit
those rows even though both GPUs are gfx942. Never rewrite `304` to `80`; these
rows were selected by tuning on the 80-CU device, then checked through the
production operator with numerical comparison.

Promoted coverage is intentionally small: C1 M=7/8, C8 M=56/64 and M=4096
prefill/throughput-profile shapes with demonstrated end-to-end value. Experimental M~1024 rows
were not promoted because they did not improve the true-cold 200K request
isolation benchmark.

AITER 0.1.19 tuner note: same-process post-build verification can retain an old
native extension registry. If a candidate compile succeeds but compare reports
`kernel ... is not present in the compiled registry`, verify the candidate in a
fresh Python process before deciding.
