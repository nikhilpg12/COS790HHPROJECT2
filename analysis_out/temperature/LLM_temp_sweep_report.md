# LLM-GP hyper-heuristic — results analysis


**5 run directories** — 5 complete, 0 mid-run, 0 empty.

## Validity gates (per run)

A failed gate means the corresponding number is **suppressed** below, not shown with a caveat.

| run | feasible | SCV vs pub | timing |
| --- | --- | --- | --- | 
| results/temperature/T_0.2/car-f-92-20260831T144121Z-seed1001 | ✅ | ✅ | ✅ | 
| results/temperature/T_0.4/car-f-92-20260831T153114Z-seed1001 | ✅ | ✅ | ✅ | 
| results/temperature/T_0.6/car-f-92-20260831T162127Z-seed1001 | ✅ | ✅ | ✅ | 
| results/temperature/T_0.8/car-f-92-20260831T171116Z-seed1001 | ✅ | ✅ | ✅ | 
| results/temperature/T_1.0/car-f-92-20260831T180221Z-seed1001 | ❌ | ❌ | ✅ | 

## Search behaviour

| run | time-to-feasible (phase) | generations regressed | Σ elitism loss (fitness) | final best = global best | parent→child fitness r | n pairs |
| --- | --- | --- | --- | --- | --- | --- |
| results/temperature/T_0.2/car-f-92-20260831T144121Z-seed1001 | 5 | 0 | 0.000 | True | 0.00 | 200 |
| results/temperature/T_0.4/car-f-92-20260831T153114Z-seed1001 | 4 | 0 | 0.000 | True | 0.17 | 200 |
| results/temperature/T_0.6/car-f-92-20260831T162127Z-seed1001 | 1 | 0 | 0.000 | True | 0.20 | 200 |
| results/temperature/T_0.8/car-f-92-20260831T171116Z-seed1001 | 4 | 0 | 0.000 | True | -0.02 | 200 |
| results/temperature/T_1.0/car-f-92-20260831T180221Z-seed1001 | never | 0 | 0.000 | True | 0.14 | 200 |

_`generations regressed` counts generations whose best (HCV, SCV) is worse than the best already seen — the direct cost of having no elitism. `parent→child fitness r` near 0 means the LLM operators are not heritable._

### results/temperature/T_0.2/car-f-92-20260831T144121Z-seed1001 — best per generation

| phase | gen best HCV | gen best SCV | so-far HCV | so-far SCV | feasible frac | elitism loss | mean nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 6 | 4.730 | 6 | 4.730 | 0.00 | 0.000 | 7.4 |
| 0 | 5 | 4.735 | 5 | 4.735 | 0.00 | 0.000 | 10.7 |
| 1 | 3 | 4.880 | 3 | 4.880 | 0.00 | 0.000 | 12.2 |
| 2 | 3 | 4.880 | 3 | 4.880 | 0.00 | 0.000 | 12.8 |
| 3 | 3 | 4.880 | 3 | 4.880 | 0.00 | 0.000 | 13.7 |
| 4 | 2 | 4.847 | 2 | 4.847 | 0.00 | 0.000 | 12.9 |
| 5 | 0 | 4.656 | 0 | 4.656 | 0.05 | 0.000 | 13.6 |
| 6 | 0 | 4.656 | 0 | 4.656 | 0.15 | 0.000 | 14.4 |
| 7 | 0 | 4.656 | 0 | 4.656 | 0.05 | 0.000 | 15.9 |
| 8 | 0 | 4.643 | 0 | 4.643 | 0.10 | 0.000 | 16.6 |
| 9 | 0 | 4.643 | 0 | 4.643 | 0.10 | 0.000 | 17.3 |

### results/temperature/T_0.4/car-f-92-20260831T153114Z-seed1001 — best per generation

| phase | gen best HCV | gen best SCV | so-far HCV | so-far SCV | feasible frac | elitism loss | mean nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 6 | 4.522 | 6 | 4.522 | 0.00 | 0.000 | 7.7 |
| 0 | 3 | 4.541 | 3 | 4.541 | 0.00 | 0.000 | 9.8 |
| 1 | 3 | 4.541 | 3 | 4.541 | 0.00 | 0.000 | 10.2 |
| 2 | 1 | 5.169 | 1 | 5.169 | 0.00 | 0.000 | 10.7 |
| 3 | 1 | 4.845 | 1 | 4.845 | 0.00 | 0.000 | 11.3 |
| 4 | 0 | 4.814 | 0 | 4.814 | 0.05 | 0.000 | 11.9 |
| 5 | 0 | 4.814 | 0 | 4.814 | 0.10 | 0.000 | 11.0 |
| 6 | 0 | 4.681 | 0 | 4.681 | 0.10 | 0.000 | 11.6 |
| 7 | 0 | 4.681 | 0 | 4.681 | 0.25 | 0.000 | 11.2 |
| 8 | 0 | 4.681 | 0 | 4.681 | 0.40 | 0.000 | 11.9 |
| 9 | 0 | 4.681 | 0 | 4.681 | 0.35 | 0.000 | 12.8 |

### results/temperature/T_0.6/car-f-92-20260831T162127Z-seed1001 — best per generation

| phase | gen best HCV | gen best SCV | so-far HCV | so-far SCV | feasible frac | elitism loss | mean nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 5 | 4.545 | 5 | 4.545 | 0.00 | 0.000 | 7.4 |
| 0 | 3 | 4.541 | 3 | 4.541 | 0.00 | 0.000 | 10.1 |
| 1 | 0 | 5.146 | 0 | 5.146 | 0.05 | 0.000 | 12.4 |
| 2 | 0 | 5.146 | 0 | 5.146 | 0.05 | 0.000 | 12.7 |
| 3 | 0 | 5.146 | 0 | 5.146 | 0.15 | 0.000 | 13.0 |
| 4 | 0 | 5.146 | 0 | 5.146 | 0.05 | 0.000 | 12.6 |
| 5 | 0 | 4.796 | 0 | 4.796 | 0.15 | 0.000 | 13.9 |
| 6 | 0 | 4.796 | 0 | 4.796 | 0.10 | 0.000 | 14.9 |
| 7 | 0 | 4.796 | 0 | 4.796 | 0.20 | 0.000 | 15.4 |
| 8 | 0 | 4.796 | 0 | 4.796 | 0.15 | 0.000 | 14.2 |
| 9 | 0 | 4.796 | 0 | 4.796 | 0.55 | 0.000 | 14.0 |

### results/temperature/T_0.8/car-f-92-20260831T171116Z-seed1001 — best per generation

| phase | gen best HCV | gen best SCV | so-far HCV | so-far SCV | feasible frac | elitism loss | mean nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 5 | 4.545 | 5 | 4.545 | 0.00 | 0.000 | 7.3 |
| 0 | 2 | 4.760 | 2 | 4.760 | 0.00 | 0.000 | 9.7 |
| 1 | 1 | 4.985 | 1 | 4.985 | 0.00 | 0.000 | 12.2 |
| 2 | 1 | 4.985 | 1 | 4.985 | 0.00 | 0.000 | 12.1 |
| 3 | 1 | 4.985 | 1 | 4.985 | 0.00 | 0.000 | 12.8 |
| 4 | 0 | 5.934 | 0 | 5.934 | 0.05 | 0.000 | 12.8 |
| 5 | 0 | 5.934 | 0 | 5.934 | 0.05 | 0.000 | 14.2 |
| 6 | 0 | 5.273 | 0 | 5.273 | 0.15 | 0.000 | 16.5 |
| 7 | 0 | 5.273 | 0 | 5.273 | 0.10 | 0.000 | 17.2 |
| 8 | 0 | 4.845 | 0 | 4.845 | 0.10 | 0.000 | 17.8 |
| 9 | 0 | 4.845 | 0 | 4.845 | 0.10 | 0.000 | 18.0 |

### results/temperature/T_1.0/car-f-92-20260831T180221Z-seed1001 — best per generation

| phase | gen best HCV | gen best SCV | so-far HCV | so-far SCV | feasible frac | elitism loss | mean nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 5 | 4.545 | 5 | 4.545 | 0.00 | 0.000 | 7.3 |
| 0 | 2 | 4.760 | 2 | 4.760 | 0.00 | 0.000 | 10.1 |
| 1 | 1 | 4.985 | 1 | 4.985 | 0.00 | 0.000 | 12.6 |
| 2 | 1 | 4.985 | 1 | 4.985 | 0.00 | 0.000 | 12.4 |
| 3 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 14.8 |
| 4 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 15.3 |
| 5 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 16.3 |
| 6 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 17.4 |
| 7 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 17.4 |
| 8 | 1 | 4.949 | 1 | 4.949 | 0.00 | 0.000 | 17.7 |
| 9 | 1 | 4.697 | 1 | 4.697 | 0.00 | 0.000 | 17.2 |

## Diversity

| run | candidates | unique tree ratio | unique canonical ratio | trees collapsed by canon | distinct (hcv,scv) | terminal KL vs uniform |
| --- | --- | --- | --- | --- | --- | --- |
| results/temperature/T_0.2/car-f-92-20260831T144121Z-seed1001 | 220 | 0.77 | 0.75 | 10 | 210 | 0.010 |
| results/temperature/T_0.4/car-f-92-20260831T153114Z-seed1001 | 220 | 0.83 | 0.83 | 0 | 208 | 0.025 |
| results/temperature/T_0.6/car-f-92-20260831T162127Z-seed1001 | 220 | 0.85 | 0.85 | 2 | 209 | 0.013 |
| results/temperature/T_0.8/car-f-92-20260831T171116Z-seed1001 | 220 | 0.79 | 0.79 | 0 | 210 | 0.038 |
| results/temperature/T_1.0/car-f-92-20260831T180221Z-seed1001 | 220 | 0.79 | 0.79 | 0 | 209 | 0.007 |

**Cross-run:** 1100 candidates over 5 runs → 770 unique raw trees, 763 unique canonical trees.

Most repeated tree: `(* (+ a b) (- c d))` — 14 occurrences.

Trees emitted in more than one run:

| tree | runs |
| --- | --- |
| (* (+ a b) (- c d)) | 5 |
| (* (+ d e) (- f g)) | 5 |
| (+ (* a b) (/ c d)) | 5 |
| (+ (* a b) (/ f g)) | 5 |
| (+ a (+ b (* c d))) | 5 |
| (+ a b) | 5 |
| (if (< e f) (* a b) (+ c d)) | 5 |
| (if (< e h) (* c d) (+ f g)) | 5 |
| (* (+ a b) (+ c d)) | 4 |
| (+ (* e f) (- g h)) | 4 |
| (if (< a b) (+ c d) (* e f)) | 4 |
| (+ (* a (+ b c)) (/ f g)) | 3 |
| (+ (+ a b) (* c (+ d e))) | 3 |
| (+ (+ e f) (* g h)) | 3 |
| (+ (/ (+ a b) c) (* f g)) | 3 |

## LLM health

| run | calls | invalid | latency p50 | latency max | p50 (no outliers) | max (no outliers) | max completion tokens | runaway share of LLM time | # runaways |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/temperature/T_0.2/car-f-92-20260831T144121Z-seed1001 | 159 | 34 | 0.45 | 14.1 | 0.45 | 14.1 | 2048 | 0.00 | 0 |
| results/temperature/T_0.4/car-f-92-20260831T153114Z-seed1001 | 145 | 20 | 0.43 | 15.1 | 0.43 | 15.1 | 2048 | 0.00 | 0 |
| results/temperature/T_0.6/car-f-92-20260831T162127Z-seed1001 | 156 | 31 | 0.45 | 0.8 | 0.45 | 0.8 | 74 | 0.00 | 0 |
| results/temperature/T_0.8/car-f-92-20260831T171116Z-seed1001 | 153 | 28 | 0.46 | 16.0 | 0.46 | 16.0 | 2048 | 0.00 | 0 |
| results/temperature/T_1.0/car-f-92-20260831T180221Z-seed1001 | 152 | 27 | 0.48 | 0.7 | 0.48 | 0.7 | 66 | 0.00 | 0 |

**Failure taxonomy (all runs):**

| category | count |
| --- | --- |
| unchanged_parent | 90 |
| depth_exceeded | 30 |
| arity_or_parse | 15 |
| runaway_unterminated_json | 5 |

## Aggregation by (instance, temperature, commit)

| instance | temp | commit | n | n complete | feasible rate | mean best feasible SCV | stdev | rankable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| car-f-92 | 0.200 | 8bfda5177179 | 1 | 1 | 1.00 | 4.643 |  | n=1 — not ranked |
| car-f-92 | 0.400 | 8bfda5177179 | 1 | 1 | 1.00 | 4.681 |  | n=1 — not ranked |
| car-f-92 | 0.600 | 8bfda5177179 | 1 | 1 | 1.00 | 4.796 |  | n=1 — not ranked |
| car-f-92 | 0.800 | 8bfda5177179 | 1 | 1 | 1.00 | 4.845 |  | n=1 — not ranked |
| car-f-92 | 1.000 | 8bfda5177179 | 1 | 1 | 0.00 |  |  | n=1 — not ranked |

_No cell has ≥2 complete runs. Nothing can be ranked — every temperature currently has a single seed (finding 7)._
