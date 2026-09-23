# Analysis of Depression Symptoms in NHANES

## Dataset

**DEMO_J.XPT:** Contains demographic information of NHANES respondents, including age, gender, poverty ratio of household income, and survey weights, among other variables.

**DPQ_J.XPT:** Contains the 9 questions of the PHQ-9 depression symptom scale, used to calculate the total score of PHQ-9 for each respondent.

**ALQ_J.XPT:** Contains information on respondents' drinking behavior, including whether they have ever drunk alcohol, the frequency of drinking in the past year, and the average amount of alcohol consumed.

**SMQ_J.XPT:** Contains information on respondents' smoking behavior, including whether they smoke, the age at which they started regular smoking, and their current smoking status.

## Code

**nhanes_analysis.R:** Merge four NHANES datasets based on respondent numbers, complete the handling of missing values, variable coding, and PHQ-9 score calculation; conduct t-tests, ANOVA, correlation analysis, and survey-weighted multiple linear regression, and establish multiple linear regression, Random Forest, and XGBoost models to compare the performance of different models in predicting PHQ-9 scores.

**check_analysis.R:** Check the data processing and modeling procedures of the main program, including PHQ-9 missing value handling, training set and test set division, training set preprocessing, evaluation index calculation, and whether there is leakage of the target variable or respondent numbers.

## Presentation

**ppt:** Introduces the research questions of the project, the data source of NHANES, variable processing, statistical analysis methods, and the main research results.
