Examples
========

This page contains practical examples of using fewlab in different scenarios.

E-commerce Product Labeling
----------------------------

Suppose you run an e-commerce platform and want to understand how user demographics relate to product preferences. You need to label products by category but have a limited budget.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from fewlab import items_to_label

   # User demographics
   users = pd.DataFrame({
       'age': [25, 34, 45, 29, 52, 38, 31, 41],
       'income': [35000, 65000, 85000, 42000, 95000, 58000, 48000, 72000],
       'urban': [1, 1, 0, 1, 0, 1, 1, 0],  # 1=urban, 0=suburban/rural
       'has_children': [0, 1, 1, 0, 1, 0, 1, 1]
   })

   # Product purchase counts (users × products)
   products = pd.DataFrame({
       'laptop': [2, 1, 0, 1, 1, 2, 0, 1],
       'smartphone': [1, 2, 1, 2, 0, 1, 2, 1],
       'tablet': [0, 1, 1, 0, 1, 1, 0, 2],
       'headphones': [3, 2, 1, 4, 0, 2, 3, 1],
       'camera': [0, 0, 2, 0, 1, 1, 0, 1],
       'gaming_console': [1, 0, 0, 2, 0, 0, 1, 0],
       'smartwatch': [1, 1, 0, 1, 0, 2, 1, 0],
       'speakers': [0, 1, 1, 1, 2, 0, 1, 1]
   })

   # Get top 4 products to label first
   priority_products = items_to_label(
       counts=products,
       X=users,
       K=4
   )

   print(f"Label these products first: {priority_products}")
   # Output might be: ['headphones', 'smartphone', 'laptop', 'smartwatch']

Content Moderation
-------------------

You're moderating user-generated content and need to prioritize which content to review for safety.

.. code-block:: python

   # User characteristics
   users = pd.DataFrame({
       'account_age_days': [30, 365, 10, 180, 90, 720, 45, 200],
       'follower_count': [50, 1200, 15, 300, 150, 2500, 80, 450],
       'verified': [0, 1, 0, 0, 0, 1, 0, 0],
       'posting_frequency': [2.1, 0.8, 5.2, 1.5, 3.1, 0.5, 4.0, 1.2]  # posts per day
   })

   # Content interaction counts (views, likes, shares)
   content_interactions = pd.DataFrame({
       'post_1': [10, 150, 5, 30, 25, 200, 15, 40],
       'post_2': [0, 80, 2, 15, 8, 120, 3, 20],
       'post_3': [20, 300, 1, 50, 40, 450, 25, 70],
       'post_4': [5, 20, 8, 10, 12, 25, 10, 15],
       'post_5': [0, 5, 15, 2, 3, 10, 8, 5],
       'post_6': [30, 180, 0, 25, 20, 220, 12, 35]
   })

   # Prioritize 3 posts for manual review
   priority_content = items_to_label(
       counts=content_interactions,
       X=users,
       K=3
   )

   print(f"Review these posts first: {priority_content}")

Website Feature Usage Analysis
------------------------------

You want to understand which website features to label by importance, given user behavior data.

.. code-block:: python

   # User characteristics
   users = pd.DataFrame({
       'session_length_min': [12, 45, 8, 25, 60, 15, 30, 40],
       'pages_per_session': [3, 12, 2, 6, 18, 4, 8, 10],
       'mobile_user': [1, 0, 1, 1, 0, 1, 0, 0],
       'returning_user': [0, 1, 0, 1, 1, 0, 1, 1]
   })

   # Feature usage counts
   feature_usage = pd.DataFrame({
       'search_bar': [5, 12, 2, 8, 15, 3, 6, 9],
       'filters': [1, 8, 0, 3, 12, 1, 4, 6],
       'sort_options': [2, 6, 1, 4, 10, 2, 3, 5],
       'user_reviews': [0, 5, 1, 2, 8, 0, 3, 4],
       'related_items': [3, 10, 2, 5, 14, 2, 7, 8],
       'wishlist': [1, 3, 0, 1, 6, 0, 2, 3],
       'chat_support': [0, 1, 1, 0, 2, 1, 0, 1],
       'newsletter_signup': [0, 1, 0, 1, 1, 0, 1, 1]
   })

   # Focus on top 4 features
   priority_features = items_to_label(
       counts=feature_usage,
       X=users,
       K=4
   )

   print(f"Analyze these features first: {priority_features}")

Comparing Different Approaches
------------------------------

You can compare the algorithmic selection with random sampling:

.. code-block:: python

   import numpy as np
   from fewlab import items_to_label

   # Your data
   users = pd.DataFrame({'age': range(20, 120), 'income': range(30000, 130000, 1000)})
   items = pd.DataFrame(np.random.poisson(3, (100, 20)),
                       columns=[f'item_{i}' for i in range(20)])

   # Algorithmic selection
   smart_selection = items_to_label(items, users, K=5)

   # Random selection for comparison
   random_selection = np.random.choice(items.columns, size=5, replace=False).tolist()

   print(f"Smart selection: {smart_selection}")
   print(f"Random selection: {random_selection}")

Iterative Labeling Strategy
----------------------------

Start small and expand based on results:

.. code-block:: python

   # Start with a small set
   initial_items = items_to_label(products, users, K=3)
   print(f"Round 1 - Label these {len(initial_items)} items: {initial_items}")

   # After labeling, you might want more
   if analysis_needs_more_precision():
       additional_items = items_to_label(products, users, K=6)
       new_items = [item for item in additional_items if item not in initial_items]
       print(f"Round 2 - Label these additional items: {new_items}")

Working with Large Datasets
----------------------------

For large datasets, you might want to sample users first:

.. code-block:: python

   # Sample users if dataset is very large
   if len(users) > 10000:
       sample_idx = np.random.choice(len(users), size=5000, replace=False)
       users_sample = users.iloc[sample_idx]
       items_sample = items.iloc[sample_idx]
   else:
       users_sample = users
       items_sample = items

   priority_items = items_to_label(items_sample, users_sample, K=20)

Error Handling
--------------

Robust error handling for real-world data:

.. code-block:: python

   def safe_item_selection(counts, features, K, ridge=None):
       """Wrapper with error handling."""
       try:
           # Check data alignment
           if not counts.index.equals(features.index):
               # Try to align by intersection
               common_idx = counts.index.intersection(features.index)
               if len(common_idx) == 0:
                   raise ValueError("No common users between datasets")
               counts = counts.loc[common_idx]
               features = features.loc[common_idx]
               print(f"Aligned datasets to {len(common_idx)} common users")

           # Check for empty data
           if counts.sum().sum() == 0:
               raise ValueError("No usage data found")

           return items_to_label(counts, features, K=K, ridge=ridge)

       except np.linalg.LinAlgError:
           print("Matrix singularity detected, adding ridge regularization")
           return items_to_label(counts, features, K=K, ridge=1e-6)
       except Exception as e:
           print(f"Error in item selection: {e}")
           # Fallback to random selection
           return np.random.choice(counts.columns, size=min(K, len(counts.columns)),
                                 replace=False).tolist()

   # Use the robust version
   selected_items = safe_item_selection(products, users, K=5)
