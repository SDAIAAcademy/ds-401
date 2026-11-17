# Follow-up task: review an uncertainty analysis

A tracked task within three weeks of the course. You choose one uncertainty analysis, from your workplace or a public source, and review it in writing with the checklist of Lab 4.4. It takes about two hours.

## What to choose

A report, paper or dashboard that estimates a quantity with uncertainty and draws a decision or a recommendation from it: a risk register, a forecast with intervals, a reliability study, a published Bayesian analysis. Use a published source or a workplace analysis you are allowed to discuss; never anything that contains personal data, and nothing classified above what your organisation allows you to share with the trainer.

## The checklist

Answer each item in one to three sentences, quoting the source where you can.

1. **Priors.** Is every prior or assumed distribution stated, with a reason? Was it checked by simulating what it predicts?
2. **Computation.** How were the numbers computed? If by sampling: chains, R-hat, effective sample sizes, divergences. If by a formula or an approximation: is it suited to the question?
3. **Structure.** Does the model suit groups with very unequal data (pooling, predictors)? What does it do to the smallest groups?
4. **Measurement.** Is the data what it claims to measure? Under-reporting, changes in recording, a proxy standing for something else?
5. **Checking.** Is the model checked against data it produced (predictive checks), and compared with an alternative on held-out prediction?
6. **Decision.** Is the decision taken from the whole distribution, with costs, the remaining risk and the value of more data, or from a point estimate?
7. **Ranking.** Which flaw would change the decision most if corrected? Rank the flaws you found by that effect.

## What to submit

A Markdown file of at most 800 words: the full reference of the source (or a description of the workplace analysis without confidential details), the seven answers and a verdict of three sentences: what the analysis supports, what it does not, and the one change that would most improve the decision. Add it to your capstone repository as `followup.md`, or submit it as your trainer directs.

## How it is read

For the use of the checklist and the ranking by effect on the decision, not for agreement with the source. A review that finds nothing to fix is acceptable when every item is answered with evidence.
