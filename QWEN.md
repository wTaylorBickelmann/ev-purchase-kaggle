# Qwen Code — executor only

EV purchase Kaggle. When autoloop invokes you, you are the **executor** on a **fresh session**.

1. Read `outputs/ITERATION_PLAN.md` + `iteration_plan.json`
2. Implement that one change in the new `exps/expNNNN/`
3. Do not full-train or Kaggle-submit
4. Stop when config/NOTES/(optional code) match the plan

Planner is a separate model. Do not re-plan the STRATEGY queue unless the plan says so.
