import pandas as pd
from scipy import stats

# Load career survey data
# Assume the data is in a CSV file called 'career_survey_data.csv'
data = pd.read_csv('career_survey_data.csv')

# Chi-square test
# Example: Testing the relationship between job satisfaction and salary level
contingency_table = pd.crosstab(data['job_satisfaction'], data['salary_level'])
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Chi-square Stat: {chi2_stat}, p-value: {p_value}, Degrees of Freedom: {dof}')

# T-tests
# Example: Comparing the mean salary between two job roles
role_a_salary = data[data['job_role'] == 'Role A']['salary']
role_b_salary = data[data['job_role'] == 'Role B']['salary']
t_stat, p_value_ttest = stats.ttest_ind(role_a_salary, role_b_salary)
print(f'T-test Stat: {t_stat}, p-value: {p_value_ttest}')
