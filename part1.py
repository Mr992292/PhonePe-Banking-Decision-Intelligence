import csv
import os
import numpy as np

with open("data/records.csv", "r") as f:
    rows = list(csv.DictReader(f))

columns = [
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score",
    "irregular_salary_flag",
    "autopay_flag",
    "loan_default",
    "monthly_spend_inr_thousands"
]

data = np.array([[float(row[col]) for col in columns] for row in rows])

X1, X2, X3, X4, X5, X6, X7, outcome, y = data.T
n = len(y)

mean_value = np.mean(y)
sd_value = np.std(y, ddof=1)
se_value = sd_value / np.sqrt(n)
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

z_stat = (mean_value - 8.0) / se_value
reject_h0 = abs(z_stat) > 1.96
n_required = int(np.ceil((1.96 * sd_value / 0.5) ** 2))

mean_X4 = np.mean(X4)
sd_X4 = np.std(X4, ddof=1)

reg_slope = np.sum((X4-mean_X4)*(y-mean_value))/np.sum((X4-mean_X4)**2)
reg_intercept = mean_value - reg_slope*mean_X4
fitted = reg_intercept + reg_slope*X4
residuals = y-fitted

tss = np.sum((y-mean_value)**2)
ess = np.sum((fitted-mean_value)**2)
rss = np.sum(residuals**2)
r2 = 1-rss/tss
rmse = np.sqrt(rss/n)
mae = np.mean(np.abs(residuals))
tss_equals_ess_plus_rss = abs(tss-ess-rss) < 1e-6*tss

z = (X4-mean_X4)/sd_X4
m, c = 0.0, 0.0

def calculate_cost(m,c):
    return np.mean((y-(m*z+c))**2)

gd_J0 = calculate_cost(m,c)
gd_costs = {}

for epoch in range(1,501):
    errors = y-(m*z+c)
    dm = -2*np.sum(z*errors)/n
    dc = -2*np.sum(errors)/n
    m -= 0.1*dm
    c -= 0.1*dc
    if epoch in [1,10,100,500]:
        gd_costs[epoch] = calculate_cost(m,c)

gd_J1, gd_J10 = gd_costs[1], gd_costs[10]
gd_J100, gd_J500 = gd_costs[100], gd_costs[500]
gd_slope = m/sd_X4
gd_intercept = c-m*mean_X4/sd_X4
gd_slope_gap = abs(gd_slope-reg_slope)

order = np.argsort(fitted,kind="stable")
lower,upper = order[:400],order[400:]
resid_sd_lower_raw = np.std(residuals[lower],ddof=1)
resid_sd_upper_raw = np.std(residuals[upper],ddof=1)
resid_sd_ratio_raw = resid_sd_upper_raw/resid_sd_lower_raw

log_X4,log_y = np.log(X4),np.log(y)
mean_log_X4,mean_log_y = np.mean(log_X4),np.mean(log_y)

loglog_slope = np.sum((log_X4-mean_log_X4)*(log_y-mean_log_y))/np.sum((log_X4-mean_log_X4)**2)
loglog_intercept = mean_log_y-loglog_slope*mean_log_X4
loglog_fitted = loglog_intercept+loglog_slope*log_X4
loglog_residuals = log_y-loglog_fitted

loglog_tss = np.sum((log_y-mean_log_y)**2)
loglog_rss = np.sum(loglog_residuals**2)
loglog_r2 = 1-loglog_rss/loglog_tss
loglog_rmse = np.sqrt(loglog_rss/n)

log_order = np.argsort(loglog_fitted,kind="stable")
log_lower,log_upper = log_order[:400],log_order[400:]
resid_sd_lower_log = np.std(loglog_residuals[log_lower],ddof=1)
resid_sd_upper_log = np.std(loglog_residuals[log_upper],ddof=1)
resid_sd_ratio_log = resid_sd_upper_log/resid_sd_lower_log

X_multi = np.column_stack([np.ones(n),log_X4,X1,X2,X3,X5,X6,X7])
beta = np.linalg.solve(X_multi.T@X_multi,X_multi.T@log_y)
multi_fitted = X_multi@beta
multi_rss = np.sum((log_y-multi_fitted)**2)
mr_r2 = 1-multi_rss/loglog_tss
mr_adj_r2 = 1-(1-mr_r2)*(n-1)/(n-8)
loglog_adj_r2 = 1-(1-loglog_r2)*(n-1)/(n-2)

results = {
"n":n,"mean_value":mean_value,"sd_value":sd_value,"se_value":se_value,
"ci95_low":ci95_low,"ci95_high":ci95_high,"z_stat":z_stat,
"reject_h0":"yes" if reject_h0 else "no","n_required":n_required,
"reg_slope":reg_slope,"reg_intercept":reg_intercept,
"tss":tss,"ess":ess,"rss":rss,
"tss_equals_ess_plus_rss":"yes" if tss_equals_ess_plus_rss else "no",
"r2":r2,"rmse":rmse,"mae":mae,
"gd_J0":gd_J0,"gd_J1":gd_J1,"gd_J10":gd_J10,
"gd_J100":gd_J100,"gd_J500":gd_J500,
"gd_slope":gd_slope,"gd_intercept":gd_intercept,
"gd_slope_gap":gd_slope_gap,
"resid_sd_lower_raw":resid_sd_lower_raw,
"resid_sd_upper_raw":resid_sd_upper_raw,
"resid_sd_ratio_raw":resid_sd_ratio_raw,
"loglog_intercept":loglog_intercept,"loglog_slope":loglog_slope,
"loglog_r2":loglog_r2,"loglog_rmse":loglog_rmse,
"resid_sd_lower_log":resid_sd_lower_log,
"resid_sd_upper_log":resid_sd_upper_log,
"resid_sd_ratio_log":resid_sd_ratio_log,
**{f"mr_beta_{i}":v for i,v in enumerate(beta)},
"mr_r2":mr_r2,"mr_adj_r2":mr_adj_r2,
"loglog_adj_r2":loglog_adj_r2
}

os.makedirs("outputs",exist_ok=True)

with open("outputs/part1_results.txt","w") as f:
    for key,value in results.items():
        if isinstance(value,str):
            formatted=value
        elif isinstance(value,(int,np.integer)):
            formatted=str(value)
        else:
            formatted=f"{value:.4f}"
        f.write(f"{key} = {formatted}\n")

print("Part 1 results successfully saved!")
print("Total results:",len(results))
