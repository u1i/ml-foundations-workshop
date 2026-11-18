# Mini-project brief

Your team takes one government case and decides whether a machine learning model should be used for it. You work in a guided notebook (no coding needed) and present your decision to the group.

## The three cases

| Case | Question | Notebook |
|---|---|---|
| A | Which benefit claims should the investigation team check? About 2% of claims are fraudulent, and the team can review 50 claims a week. | [Flagging fraud in benefit claims](https://colab.research.google.com/github/u1i/ml-foundations-workshop/blob/main/notebooks/09_mini_project_fraud.ipynb) |
| B | Which visa and residency applications will be late? | [Predicting late applications](https://colab.research.google.com/github/u1i/ml-foundations-workshop/blob/main/notebooks/10_mini_project_late_applications.ipynb) |
| C | Which unit should handle a citizen request written in Arabic or English? | [Routing citizen requests](https://colab.research.google.com/github/u1i/ml-foundations-workshop/blob/main/notebooks/11_mini_project_request_routing.ipynb) |

Suggested split for five tables: tables 1 and 2 take case A, tables 3 and 4 take case C, table 5 takes case B. You can swap, as long as every case is taken by at least one table.

All data is synthetic.

## Roles

- One person leads the business question: what decision should the model support, and for whom?
- One person drives the notebook on their laptop while the team decides together.
- One person prepares and gives the presentation.

## Working through the notebook

Click the play button on each step, in order, starting with Setup.

1. **The data.** Look at the first results. What do you notice?
2. **Build your model.** Try at least two changes on the right (for example remove an input, switch the model or the method) and note which version works best and why.
3. **Where does it fail?** Answer the questions in the box with your team.
4. **Explain one case.** Pick one prediction and find out why the model decided it that way. In case C, type your own request in Arabic or English.
5. **Team summary.** Type one sentence for each of the six points.

## What your team brings to the presentation

1. **Six sentences,** one for each point:
   1. What the model does
   2. How well it works on test data
   3. Where it fails
   4. The value for service users
   5. Where a person reviews the output
   6. Your recommendation: use it, improve it, or stop
2. **One number from your own results,** for example "it finds 63% of the fraud at 50 claims a week".
3. **One example case** you looked at, and why the model decided it that way.

## The presentation

One person stands up at the table and presents in 2 to 3 minutes, using the team summary from step 5. No slides are needed. The six points are shown on the main screen while teams present.

## How the presentations are scored

Five criteria, each scored from 1 to 4 (1 not addressed, 2 partly, 3 addressed, 4 addressed and backed by the team's own results). Maximum 20 points.

1. **Problem and approach:** a clear question; machine learning, generative AI or both, and why; the target defined.
2. **Data preparation:** split before preparation; missing values and categories handled; leakage checked.
3. **Model and evaluation:** compared with a simple baseline; a metric that fits the problem.
4. **Explanation and limits:** the decision explained in plain language; where the model fails.
5. **Human review and recommendation:** where a person checks the output; value for service users; a clear recommendation.

## If the notebook does not run

Ask a volunteer or the trainer for help. If it still does not work, answer the six points from your team's discussion and from what the trainer shows on screen. Every team presents.
