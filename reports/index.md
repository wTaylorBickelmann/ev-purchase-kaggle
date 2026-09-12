# Experiment index

Short table only. **Do not rewrite old rows.** Newest at bottom.

| id | idea | CV | LB | verdict | notes |
|----|------|----|----|---------|-------|
| exp0000 | Deotte Fable 5.1 3×XGB equal blend (baseline) | 0.94210 ± 0.00075 | 0.94182 | **keep** | floor for loop; seeds=42 folds=5 |
| exp0001 | deotte-baseline (no change) | 0.94210 ± 0.00075 | — | kill | train_args identical to parent |
| exp0001 | deotte-3seed-blend | 0.94212 ± 0.00075 | — | kill | best=0.94210 parent=exp0000 |
| exp0001 | deotte-plus-lgbm-m4 | — | — | kill | exec: executor timed out after 3600s |
| exp0001 | deotte-freq-income-commute | 0.94333 ± 0.00070 | pending | **keep** | parent=exp0000 |
| exp0002 | deotte-te-income | — | — | kill | exec: executor timed out after 3600s |
| exp0002 | deotte-freq2-age-nmode | 0.94328 ± 0.00073 | — | kill | best=0.94333 parent=exp0001 |
| exp0002 | deotte-te-income-commute | — | — | kill | exec: executor timed out after 3600s |
| exp0008 | deotte-plus-lgbm-m4 | 0.94336 ± 0.00070 | — | kill | best=0.94333 parent=exp0001 |
| exp0009 | deotte-blend-weight-search | 0.94333 ± 0.00070 | — | kill | best=0.94333 parent=exp0001 |
| exp0010 | deotte-te-income | 0.94552 ± 0.00064 | pending | **keep** | parent=exp0001 |
| exp0011 | deotte-te-commute | 0.94557 ± 0.00061 | — | kill | best=0.94552 parent=exp0010 |
| exp0012 | deotte-te-m2 | 0.94559 ± 0.00065 | — | kill | best=0.94552 parent=exp0010 |
| exp0013 | deotte-lr02 | 0.94555 ± 0.00064 | — | kill | best=0.94552 parent=exp0010 |
| exp0014 | planner-fail | — | — | kill | Cursor planner timed out after 900s |
| exp0015 | deotte-te-pair | 0.94551 ± 0.00063 | — | kill | best=0.94552 parent=exp0010 |
| exp0016 | deotte-orig-te | 0.94553 ± 0.00063 | — | kill | best=0.94552 parent=exp0010 |
| exp0017 | planner-fail | — | — | kill | Cursor planner timed out after 900s |
| exp0018 | planner-fail | — | — | kill | Cursor planner exited 1; see /Users/will/Documents/code_proj |
