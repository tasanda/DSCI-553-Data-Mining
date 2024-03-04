# Method Description: The recommendation system integrates various data sources to predict user-business ratings, employing XGBoost for regression and a switching hybrid method. Data from JSON files like business, check-in, photo, tip, and user files is processed for features like business attributes, check-in counts, and user interactions. The system utilizes Optuna for XGBoost hyperparameter tuning, enhancing accuracy through feature-engineered data.

# A hybrid approach combines item-based and model-based predictions, favoring item-based predictions when both user and business are in the training set. It optimizes predictions based on their presence in the data. In a switching strategy, it exclusively employs item-based ratings for pairs found in the training data, ensuring better accuracy for well-represented pairs. For absent pairs, the model resorts to model-based ratings, catering to less common or new pairs.

# This strategy offers specialized predictions based on training data availability, enhancing accuracy for known pairs while leveraging a model-based approach for less common ones. It iterates through user-business pairs, switching methods based on their presence in the training set. This adaptive approach refines predictions, relying on explicit data when available and transitioning to a model-based approach for unseen or less represented pairs, potentially improving recommendation accuracy.

# Error Distribution:
# >=0 and <=1: 102311
# >=1 and <=2: 32746
# >=2 and <=3: 6145
# >=3 and <=4: 840
# >=4: 2

# RMSE: 0.9793315284893849

# Execution Time: 377.43919038772583
