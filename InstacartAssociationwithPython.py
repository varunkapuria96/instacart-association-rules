# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the data from InstacartTransactions.csv
instacartTransactions = pd.read_csv("InstacartTransactions.csv")

# Display summary of the transactions
print(instacartTransactions.head())

# Transform the data into a one-hot encoded format
instacart_onehot = pd.get_dummies(instacartTransactions, columns=['ItemID'])

# Group by OrderID and aggregate the items purchased
instacart_basket = instacart_onehot.groupby(['OrderID']).sum()

# Convert item counts to binary values (1 if item was purchased, 0 otherwise)
instacart_basket = instacart_basket.applymap(lambda x: 1 if x > 0 else 0)

# Find item frequencies
item_frequencies = instacart_basket.mean()

# Display item frequencies
print(item_frequencies)

# Generate frequent item sets
frequent_itemsets = apriori(instacart_basket, min_support=0.005, use_colnames=True)

# Generate association rules
association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)

# Display summary of association rules
print(association_rules_df.head(10))

# Sort association rules by lift and display the top 10
association_rules_df = association_rules_df.sort_values(by='lift', ascending=False)
print(association_rules_df.head(10))
