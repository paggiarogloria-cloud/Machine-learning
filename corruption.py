# %% [markdown]
# # Causal Analysis of Firm Growth on Corruption in Vietnam
# 
# ## Research Overview
# This project examines the causal relationship between firm employment growth and corruption levels in Vietnamese firms, replicating and extending [Bai et al. (2019)](https://academic.oup.com/ej/article/129/618/651/5290389?login=false) using Double Machine Learning methods.
# 
# ## Core Research Question
# **Does firm growth reduce corruption burdens?**  
# We investigate whether increasing employment leads to decreased bribery demands as a percentage of firm revenue.
# 
# ## Empirical Framework
# 
# ### Variable Specification
# | Role                  | Variable                     | Description                          |
# |-----------------------|------------------------------|--------------------------------------|
# | **Dependent**         | `bribe_pctrev_rjt`           | Bribes as % of firm revenue         |
# | **Treatment**         | `lntotalemploy_rjt`          | Log total employment (Vietnam)      |
# | **Instrument**        | `lnchtotalemploy_jt`         | Log industry employment (China)     |
# 
# ## Instrumental Variable Approach to Address Endogeneity
# 
# To address potential endogeneity concerns in employment (e.g., reverse causality or omitted variables), we implement an instrumental variable (IV) strategy,leveraging employment trends in corresponding Chinese industries as an exogenous source of variation.
# 
# ## Characteristics of the dataset
# 
# The dataset, drawn from Bai et al. (2019), includes over 20,000 firm-year observations across Vietnamese provinces. It captures detailed firm-level information on bribes, size, ownership, and geographic characteristics, allowing us to study how firm growth affects corruption dynamics.

# %%
import pandas as pd
df = pd.read_stata("finalanalysissample.dta")
# Subset the data
dfvip = df[[
    'id',
    'year',
    'pci_id',
    'prov',
    'bribe_pctrev',
    'yrsopen',
    'employ_n',
    'meanemploy_jt',
    'lntotalemploy_jt',
    'lnprem',
    'ownland',
    'lurc',
    'ownlandnlc',
    'numdocs',
    'opdich',
    'sharedocs',
    'formerhhfirm',
    'formerSOE',
    'ownergov',
    'govholdshare',
    'lntotalemploy_r_jt',
    'lnchtotalemploy_jt',  
    'bribe_pctrev_rjt',
    'lntotalemploy_rjt',
]]
print(dfvip)

# %% [markdown]
# # **Descriptive Statistics**
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
summary_vars = ['bribe_pctrev_rjt', 'lntotalemploy_rjt', 'lnchtotalemploy_jt']
summary_stats = df[summary_vars].describe()
print("Summary Statistics:\n", summary_stats)

# %% [markdown]
# # Bribery Burden Distribution Analysis
# 
# ## Distribution Characteristics
# - **Shape**: Highly right-skewed distribution
# - **Central Tendency**:
#   - Peak concentration: 2-3% of revenue
#   - Majority of firms: <5% bribe burden
# - **Outliers**:
#   - Extreme cases reaching up to 35%
#   - Long right tail indicates significant variation
# - **This suggests that while corruption is widespread, most firms face relatively small burdens, and a few outliers experience very high corruption levels.**

# %%
sns.histplot(df['bribe_pctrev_rjt'], kde=True)
plt.title("Distribution of Corruption (% of revenue)")
plt.xlabel("bribe_pctrev_rjt")
plt.show()

# %% [markdown]
# # Firm Size Distribution Analysis
# 
# ## Distribution Characteristics
# - **Shape**: Moderately right-skewed
# - **Central Tendency**:
#   - Peak concentration: log values of 8–9
#   - Corresponds to 3,000–8,000 employees (exponentiated)
# - **Variation**:
#   - Long right tail indicates some very large firms
#   - Overall spread remains moderate
# - **This pattern supports meaningful variation in firm size across regions, which is useful for identifying its relationship with corruption.**

# %%
sns.histplot(df['lntotalemploy_rjt'], kde=True, color='green')
plt.title("Distribution of Firm Size (log total employment in region)")
plt.xlabel("lntotalemploy_rjt")
plt.show()

# %% [markdown]
# # Instrument Variable Distribution Analysis
# 
# ## Histogram with KDE Visualization
# - **Variable**: Logarithm of total employment in China (region-adjusted)
# - **Distribution Shape**: Right-skewed with distinct peaks
# - **Multiple smaller peaks suggest the presence of distinct clusters or subgroups within the data.**

# %%
sns.histplot(df['lnchtotalemploy_jt'], kde=True, color='orange')
plt.title("Distribution of Instrument (log China total employment excluding region)")
plt.xlabel("lnchtotalemploy_jt")
plt.show()

# %% [markdown]
# # Corruption and Firm Size: A Quantile Analysis  
# 
# This code segments firms into **four size quantiles (Q1–Q4)** using the logarithm of total employment (`lntotalemploy_rjt`), then computes the mean corruption level (`bribe_pctrev_rjt`) per group.  
# 
# **Key Insight**:  
# - **Q1 (Smallest firms)**: Avg. corruption = **3.94**  
# - **Q4 (Largest firms)**: Avg. corruption = **2.90**  
# 
# The inverse relationship suggests **larger firms report systematically lower corruption**, hinting at economies of scale in compliance or bargaining power against rent-seeking. 

# %%
df['firm_size_quantile'] = pd.qcut(df['lntotalemploy_rjt'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
grouped_means = df.groupby('firm_size_quantile')['bribe_pctrev_rjt'].mean()
print("Average Corruption by Firm Size Quantile:\n", grouped_means)

# %% [markdown]
# ## Instrument Validity & Preliminary Relationships
# 
# ### Observed Correlations
# 
# **Core Relationship**  
# We observe a **negative correlation (-0.15)** between firm size (`lntotalemploy_rjt`) and reported corruption (`bribe_pctrev_rjt`), suggesting larger firms tend to experience slightly lower bribery burdens. While modest in magnitude, this directional relationship aligns with theoretical expectations about firm size and corruption vulnerability.
# 
# **Instrument Diagnostics**  
# The instrument (`lnchtotalemploy_jt`) shows:
# - Strong **relevance** (r = 0.56) with the endogenous treatment variable  
# - Moderate but acceptable **exogeneity** (r = -0.39) with the outcome  

# %%
correlation = df[summary_vars].corr()
print("Correlation Matrix:\n", correlation)

# %%
print(dfvip.columns.tolist())

# %%
pct_null = dfvip.isnull().sum() / len(dfvip)
print(pct_null)

# %%
dfvip.year.unique()

# %%
dfvip['year'].value_counts()

# %%
import pandas as pd
import numpy as np
from sklearn import tree
from doubleml import DoubleMLData, DoubleMLPLR

# %%
import pyfixest as pf 

# %%
print("prov" in df.columns) 
df["prov"] = pd.Categorical(df["prov"])

# %%
m1 = pf.feols(
    "bribe_pctrev_rjt ~ lnchtotalemploy_jt + lntotalemploy_rjt + yrsopen + employ_n + meanemploy_jt + lnprem + ownland + lurc + ownlandnlc + numdocs + opdich + sharedocs + formerhhfirm + formerSOE + ownergov + govholdshare  | year + prov",
    vcov = {"CRV1": "prov", "CRV2": "year"},  # Single dictionary for multi-way clustering
    data=df
)
m2 = pf.feols(
    "bribe_pctrev_rjt ~ yrsopen + employ_n + meanemploy_jt + lnprem + ownland + lurc + ownlandnlc + numdocs + opdich + sharedocs + formerhhfirm + formerSOE + ownergov + govholdshare | prov + year | lntotalemploy_rjt ~ lnchtotalemploy_jt",
    vcov = {"CRV1": "prov", "CRV2": "year"},  # Single dictionary for multi-way clustering
    data=df
)
pf.etable([m1,m2])

# %% [markdown]
# # Employment and Corruption Analysis: OLS vs IV Approaches
# 
# ## Initial OLS Results with Fixed Effects
# Our first specification uses OLS regression with **year and province fixed effects** to examine the relationship between firm employment and bribe payments:
# 
# - **Unexpected finding**: A 1% increase in total employment is associated with a **0.37 percentage point increase** in bribes (contrary to theoretical expectations)
# 
# ## Endogeneity Concerns
# The employment variable (measured in levels) may suffer from two key sources of endogeneity:
# 
# 1. **Reverse causality**: Firms in less corrupt provinces may find it easier to:
#    - Expand operations
#    - Hire more workers
#    - Grow their business
# 
# 2. **Omitted variable bias**: Unobserved factors like:
#    - Province-level business-friendly policies
#    - Institutional quality
#    - Market conditions
#    could simultaneously affect both firm growth and corruption levels
# 
# ## Instrumental Variables Approach
# To address these concerns, we implement an IV strategy using:
# 
# **Instrument**: Employment in the same industry in China for:
# - **Relevance**: Industries in both countries face:
#   - Similar global demand shocks
#   - Common price fluctuations
#   - Shared technology trends
# - **Validity**: Satisfies exclusion restriction because:
#   - China's much larger economy makes reverse causality unlikely
#   - Vietnamese corruption unlikely to affect Chinese employment
# 
# ## IV Regression Results
# After instrumenting, we find:
# - A 1% increase in total employment leads to a **0.9 percentage point decrease** in bribe payments
# - Model includes:
#   - Province fixed effects (controls for regional heterogeneity)
#   - Year fixed effects (accounts for national time trends)
# 
# **Interpretation**:
# - The IV estimate shows the expected negative relationship
# - Magnitude is smaller but directionally consistent with prior literature
# - Results provide more credible causal evidence than OLS

# %%
years = [2006, 2007, 2008, 2009, 2010]

# Create each dummy variable
for year in years:
    dfvip[f'year_{year}'] = (dfvip['year'] == year).astype(int)

# %%
dfvip = pd.get_dummies(dfvip, columns=['prov'], prefix='prov')

print(dfvip.head())

# %%
# DATA PREPARATION
# 1. Load and clean data
dfvip_clean = dfvip[[
    'id', 'year', 'pci_id', 'bribe_pctrev', 'yrsopen',
    'employ_n', 'meanemploy_jt', 'lntotalemploy_jt', 'lnprem',
    'ownland', 'lurc', 'ownlandnlc', 'numdocs', 'opdich',
    'sharedocs', 'formerhhfirm', 'formerSOE', 'ownergov',
    'govholdshare', 'lntotalemploy_r_jt', 'lnchtotalemploy_jt',
    'bribe_pctrev_rjt', 'lntotalemploy_rjt', 'year_2007', 'year_2008',
    'year_2009', 'year_2010', 'year_2006'
] + [col for col in dfvip.columns if col.startswith('prov_')]].copy()


# %%
binary_cols = [
    'ownland', 'opdich', 'sharedocs', 'formerhhfirm',
    'formerSOE', 'ownergov', 'govholdshare'
]+ [col for col in dfvip.columns if col.startswith('prov_')]

for col in binary_cols:
    dfvip_clean[col] = (
        dfvip_clean[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map({'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0})
        .fillna(0)
        .astype(int)
    )


# %%
numeric_cols = [
    'lntotalemploy_r_jt', 'yrsopen', 'employ_n', 'meanemploy_jt',
    'lnprem', 'lurc', 'ownlandnlc', 'numdocs', 'lnchtotalemploy_jt',
    'bribe_pctrev_rjt', 'lntotalemploy_rjt'
]

for col in numeric_cols:
    dfvip_clean[col] = pd.to_numeric(dfvip_clean[col], errors='coerce')

dfvip_clean = dfvip_clean.dropna(subset=numeric_cols)


# %% [markdown]
# # **DML Model Setup: Variable Analysis**  
# 
# ## **1. Variable Definitions**  
# 
# ### **Outcome Variable (`y_col`)**  
# - **`bribe_pctrev_rjt`**: Percentage of firm revenue paid as bribes (dependent variable).  
# 
# ### **Treatment Variable (`d_cols`)**  
# - **`lntotalemploy_rjt`**: Natural log of total employment (primary treatment effect of interest).  
# 
# ### **Control Variables (`x_cols`)**  
# #### **Firm Characteristics**  
# - `yrsopen`: Years since firm opening.  
# - `employ_n`, `meanemploy_jt`: Employee count and averages.  
# - `lnprem`: Log value of firm premises.  
# - Land/ownership dummies (`ownland`, `lurc`, `ownlandnlc`).  
# - Regulatory metrics (`numdocs`, `opdich`, `sharedocs`).  
# 
# #### **Ownership & Governance**  
# - `formerhhfirm`/`formerSOE`: Former household or state-owned status.  
# - `ownergov`/`govholdshare`: Government ownership stakes.  
# 
# #### **Fixed Effects**  
# - Year dummies (`year_2006`–`year_2010`).  
# - Province dummies (`prov_*`).  
# 
# ## **2. DoubleML Data Structure**  
# Initializes a `DoubleMLData` object for causal inference, isolating treatment effects while controlling for confounders.  

# %%
# DML MODEL SETUP

# Define variables
y_col = 'bribe_pctrev_rjt'  # Outcome
d_cols = 'lntotalemploy_rjt'  # Treatment
x_cols = [
    'yrsopen', 'employ_n', 'meanemploy_jt',
    'lnprem', 'ownland', 'lurc', 'ownlandnlc', 'numdocs',
    'opdich', 'sharedocs', 'formerhhfirm', 'formerSOE',
    'ownergov', 'govholdshare','year_2008','year_2009','year_2010','year_2006', 'year_2007'
]+ [col for col in dfvip.columns if col.startswith('prov_')]

# Initialize DoubleMLData
dml_data = DoubleMLData(
    data=dfvip_clean,
    y_col=y_col,
    d_cols=d_cols,
    x_cols=x_cols
)

# %% [markdown]
# # **Model Training: Double Machine Learning (DML) Implementation**
# 
# ## **1. Learner Initialization**
# - **Base Learners**: 
#   - Both the outcome model (`ml_l`) and treatment model (`ml_m`) are **Decision Tree Regressors** with fixed random state for reproducibility.
#   - Decision trees capture non-linear relationships while avoiding parametric assumptions.
# 
# ## **2. Model Specification**
# - **DoubleMLPLR**: 
#   - Implements **Partially Linear Regression (PLR)** framework.
#   - Separately models:
#     - Outcome (`y ~ d + x`)
#     - Treatment (`d ~ x`)
# 
# ## **3. Hyperparameter Tuning**
# - **Tuning Strategy**:
#   - Grid search over 100 log-spaced `ccp_alpha` values (complexity parameter for pruning).
#   - 5-fold cross-validation to prevent overfitting.
# 
# ## **4. Model Fitting**
# - Final estimation of causal effects after tuning.

# %%
# Initialize learners
ml_l = tree.DecisionTreeRegressor(random_state=123)
ml_m = tree.DecisionTreeRegressor(random_state=123)

# Initialize PLR model
dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m)

# Hyperparameter tuning
param_grids = {
    'ml_l': {'ccp_alpha': np.logspace(-2, 2, 100)}, 
    'ml_m': {'ccp_alpha': np.logspace(-2, 2, 100)}
}
np.random.seed(123)
dml_plr.tune(param_grids, search_mode='grid_search', n_folds_tune=5)

# Fit model
dml_plr.fit()


# %%
print(dml_plr.fit())

# %%
print(dml_plr.params) 

# %% [markdown]
# # **Decision Tree Hyperparameter Tuning**
# 
# ## **1. Purpose**
# - Finds optimal complexity parameters (`ccp_alpha`) for:
#   - **Outcome model** (`ml_l`): Predicting `bribe_pctrev_rjt`
#   - **Treatment model** (`ml_m`): Predicting `lntotalemploy_rjt`
# - Uses cost-complexity pruning to prevent overfitting.
# 
# ## **2. Methodology**
# 1. **Pruning Path Calculation**:
#    - Computes possible `ccp_alpha` values using `cost_complexity_pruning_path`
# 2. **Grid Search**:
#    - 5-fold cross-validation to evaluate each alpha
#    - Selects alpha that minimizes validation error
# ## **3. Output**
# - Prints optimal `ccp_alpha` for each model
# - These values should be used in final model training

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

X = dfvip_clean[x_cols]
y = dfvip_clean[y_col]

regr = DecisionTreeRegressor(random_state=123)
path = regr.cost_complexity_pruning_path(X, y)
path=pd.DataFrame(path)
aa = np.array(path['ccp_alphas'])


grid_params = {'ccp_alpha': aa}

cv_regr = DecisionTreeRegressor(random_state=123)
cv = GridSearchCV(cv_regr, grid_params, cv=5, refit=True)
cv.fit(X, y)
best_alpha_ml_l = cv.best_params_['ccp_alpha']
print(f'Best alpha for mL_l: {best_alpha_ml_l}')

# Tune mL_m (treatment model: lntotalemploy_rjt)
y = dfvip_clean[d_cols]
regr = tree.DecisionTreeRegressor(criterion='squared_error', random_state=123)
path = regr.cost_complexity_pruning_path(X,y)
path=pd.DataFrame(path)

aa = np.array(path['ccp_alphas'])
grid_params = {
    'ccp_alpha': aa}

cv_regr = DecisionTreeRegressor(random_state=123)
cv = GridSearchCV(cv_regr, grid_params, cv=5, refit=True)
cv.fit(X, y)
best_alpha_ml_m = cv.best_params_['ccp_alpha']
print(f'Best alpha for mL_m: {best_alpha_ml_m}')

# %%
pip install doubleml

# %%
best_alpha_ml_m = cv.best_params_
best_alpha_ml_l = cv.best_params_

# %%
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pyfixest as pf # 
from linearmodels.iv import IV2SLS 
from stargazer.stargazer import Stargazer
import doubleml # 
from sklearn import tree
from sklearn.model_selection import GridSearchCV # for CV grid search
#from sklearn.model_selection import
from causaldata import social_insure # pip install causaldata (package with data)

# %%
ml_l = tree.DecisionTreeRegressor(criterion='squared_error', ccp_alpha=best_alpha_ml_l['ccp_alpha'], random_state=123)
ml_m = tree.DecisionTreeRegressor(criterion='squared_error', ccp_alpha=best_alpha_ml_m['ccp_alpha'], random_state=123)
np.random.seed(123) # we need to set a seed otherwise we will get different results each time we run the next command due to the CV
dml_plr = doubleml.DoubleMLPLR(dml_data,
                            ml_l = ml_l,
                            ml_m = ml_m)
print(dml_plr.fit())

# %% [markdown]
# ## Estimation Results
# 
# ### Treatment Effect Analysis
# | Parameter          | Estimate    | Std. Error | p-value | 95% CI            |
# |--------------------|-------------|------------|---------|-------------------|
# | Firm Size Effect   | 0.000172    | 0.016587   | 0.992   | [-0.032, 0.033]   |
# 
# ### Model Diagnostics
# | Model Component     | Performance (RMSE) |
# |--------------------|--------------------|
# | Outcome Prediction | 2.275             |
# | Treatment Prediction| 1.631            |
# 
# ## Substantive Interpretation
# 
# ### Key Conclusions
# - **Null Finding**: No detectable relationship between firm size and bribery incidence
# - **Effect Magnitude**: The estimated coefficient (0.0002) suggests that even if an effect exists, it is economically negligible
# - **Precision**: Tight confidence interval around zero indicates high estimation precision

# %%
dml_plr.set_ml_nuisance_params('ml_l', 'lntotalemploy_rjt', {'ccp_alpha': best_alpha_ml_l['ccp_alpha']})
dml_plr.set_ml_nuisance_params('ml_m', 'lntotalemploy_rjt', {'ccp_alpha': best_alpha_ml_m['ccp_alpha']})

# %% [markdown]
# 
# # Instrumental Variable Double Machine Learning Analysis
# 
# ## Model Specification
# 
# ### Core Components
# | Component       | Variable               | Description                          | 
# |-----------------|------------------------|--------------------------------------|
# | **Outcome (Y)** | `bribe_pctrev_rjt`     | Bribery as % of revenue              |
# | **Treatment (D)**| `lntotalemploy_rjt`   | Log of total employment in Vietnam   |
# | **Instrument (Z)**| `lnchtotalemploy_jt` | Log of total employment in China     | 
# 
# ### Control Variables
# **Firm Characteristics:**
# - Size: `employ_n`, `meanemploy_jt`
# - Assets: `lnprem`, `ownland`
# - Ownership: `formerSOE`, `govholdshare`
# 
# **Fixed Effects:**
# - Year: 2006-2010 indicators
# - Province: 56 regional indicators
# 

# %%
obj_dml_data = doubleml.DoubleMLData(
    dfvip_clean, y_col='bribe_pctrev_rjt', d_cols='lntotalemploy_rjt',
    z_cols='lnchtotalemploy_jt', x_cols=['yrsopen', 'employ_n', 'meanemploy_jt',
    'lnprem', 'ownland', 'lurc', 'ownlandnlc', 'numdocs',
    'opdich', 'sharedocs', 'formerhhfirm', 'formerSOE',
    'ownergov', 'govholdshare','year_2008','year_2009','year_2010','year_2006', 'year_2007']+ [col for col in dfvip.columns if col.startswith('prov_')])
# specify the functions:
ml_l = tree.DecisionTreeRegressor(criterion='squared_error', ccp_alpha=best_alpha_ml_l['ccp_alpha'], random_state=123)
ml_m = tree.DecisionTreeRegressor(criterion='squared_error', random_state=123) 
ml_r = tree.DecisionTreeRegressor(criterion='squared_error', ccp_alpha=best_alpha_ml_m['ccp_alpha'], random_state=123)
np.random.seed(123) # we need to set a seed otherwise we will get different results each time we run the next command due to the CV
dml_ivm = doubleml.DoubleMLPLIV(obj_dml_data, ml_l=ml_l, ml_m= ml_m, ml_r=ml_r)
print(dml_ivm.fit())

# %% [markdown]
# # **Initial IV Results Interpretation**
# 
# ## **Key Observations**
# - **Weak causal estimate**: The initial IV coefficient of 7.65 (p=0.754) suggests:
#   - No statistically significant relationship between firm size (`lntotalemploy_rjt`) and bribery percentage
#   - Extremely wide CI [-40.29, 55.59] indicates unreliable identification

# %% [markdown]
# # **Decision Tree Hyperparameter Tuning for IV Treatment Model**
# 
# ## **Purpose**
# This code performs cost-complexity pruning to optimize the decision tree used in:
# - The **treatment model (ml_m)** of our DoubleML IV estimator
# - Goal: Find the optimal tree complexity that balances bias and variance when predicting the endogenous treatment variable (`lntotalemploy_rjt`)

# %%
y = dfvip_clean['lnchtotalemploy_jt']
dectre = ml_m.fit(X,y)
path = dectre.cost_complexity_pruning_path(X,y)
path=pd.DataFrame(path)
aa = np.array(path['ccp_alphas'])
grid_params = {
    'ccp_alpha': aa}

cv_tree = tree.DecisionTreeRegressor(criterion='squared_error', random_state= 123)
cv_tree1 = GridSearchCV(cv_tree, grid_params, cv = 5, refit=True)
cv_tree1.fit(X, y)
bestalpha_ml_m = cv_tree1.best_params_

ml_m = tree.DecisionTreeRegressor(criterion='squared_error', ccp_alpha=bestalpha_ml_m['ccp_alpha'], random_state=123) # our instrument is binary so decision trees

np.random.seed(123) # we need to set a seed otherwise we will get different results each time we run the next command due to the CV
dml_ivm = doubleml.DoubleMLPLIV(obj_dml_data, ml_l=ml_l, ml_m= ml_m, ml_r=ml_r)
print(dml_ivm.fit())

# %% [markdown]
# # **Final IV Regression Results Analysis**
# 
# ## **Key Causal Finding**
# - **Statistically significant negative effect**: 
#   - Each 1% increase in firm size (`lntotalemploy_rjt`) is associated with **1.116 percentage point decrease** in bribery as % of revenue
#   - Significant at 1% level (p = 0.0057) 
#   - 95% CI: [-1.908, -0.324] excludes zero
# 
# - **Hypothesis Confirmation**:
#    Results validate our initial hypothesis that larger firms experience lower bribery burdens as percentage of revenue
# - Consistent with theories of:
#     - Greater bargaining power against corrupt officials
#     - Economies of scale in compliance systems
#     - Reduced reliance on informal networks


