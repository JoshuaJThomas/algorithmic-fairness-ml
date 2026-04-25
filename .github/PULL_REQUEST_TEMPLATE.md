## Summary

What does this PR change and why?

## Type of change

- [ ] Bug fix in a notebook or result CSV
- [ ] New analysis or figure
- [ ] Documentation update
- [ ] Dependency or environment change
- [ ] Other

## Checklist

- [ ] My notebook changes follow the section structure in `group_notebook_contract.md`
- [ ] Model labels match exactly: `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`
- [ ] Any new CSV output uses the column schema defined in the contract
- [ ] `RANDOM_STATE = 42` and stratified 80/20 split are unchanged
- [ ] Figures are exported to `paper/figures/` using the `mX_` naming convention
- [ ] `RESULTS.md` is updated if any metrics changed
