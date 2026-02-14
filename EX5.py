import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Synthetic Historical Data
# -----------------------------
np.random.seed(42)

n_samples = 200
price = np.random.uniform(10, 100, n_samples)
season = np.random.randint(0, 2, n_samples)  # 0 = normal, 1 = peak

# True demand model (unknown to agent)
true_demand = 200 - 2.5*price + 30*season + np.random.normal(0, 10, n_samples)

data = pd.DataFrame({
    'Price': price,
    'Season': season,
    'Demand': true_demand
})

# -----------------------------
# 2. Train Predictive Model
# -----------------------------
X = data[['Price', 'Season']]
y = data['Demand']

model = LinearRegression()
model.fit(X, y)

print("Learned Coefficients:")
print("Intercept:", model.intercept_)
print("Price Coef:", model.coef_[0])
print("Season Coef:", model.coef_[1])

# -----------------------------
# 3. Model-Based Policy Optimization
# -----------------------------
def optimize_price(season_value):
    possible_prices = np.linspace(10, 100, 100)
    best_price = None
    best_revenue = -np.inf
    
    for p in possible_prices:
        predicted_demand = model.predict([[p, season_value]])[0]
        revenue = p * predicted_demand
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = p
            
    return best_price, best_revenue

# -----------------------------
# 4. Dynamic Pricing Simulation
# -----------------------------
optimal_prices = []
revenues = []

for season_value in [0, 1]:  # Normal and Peak
    opt_price, opt_revenue = optimize_price(season_value)
    optimal_prices.append(opt_price)
    revenues.append(opt_revenue)

print("\nOptimal Price (Normal Season):", optimal_prices[0])
print("Expected Revenue (Normal):", revenues[0])

print("\nOptimal Price (Peak Season):", optimal_prices[1])
print("Expected Revenue (Peak):", revenues[1])

# -----------------------------
# 5. Visualization
# -----------------------------
plt.bar(["Normal Season", "Peak Season"], optimal_prices)
plt.title("Optimal Prices by Season")
plt.ylabel("Price")
plt.show()
