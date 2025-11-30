# Notes

### How the classification workflow goes

how it produces that answer:

classify_learning_strategy (lines 482-531):

Uses rule-based if-else logic (not machine learning)
Returns the final strategy classification
This IS the answer for each player
Then later, train_strategy_tree (lines 937-984):

Takes those rule-based labels as "ground truth"
Trains an ML decision tree to replicate your rules
Used for validation/comparison with your manual rules
So you have two ways to get the final answer:

Direct: Use classify_learning_strategy (rule-based) ✓ This is the primary method
ML comparison: Use train_strategy_tree predictions (trained on rule-based labels)
In your main workflow (lines 1087-1092), classify_learning_strategy is called for each player and stored in the learning_strategy column - that's the final answer. The ML tree is optional, used mainly to see if a decision tree can learn your rules from the data.


### Current Dominance Calculation:


- For shared parameters (like frequency), the system tracks which user "owns" each time point in user_at_time
- When a user changes the frequency, they become the owner from that moment until another user changes it
- Dominance is calculated by counting how many timeline points (at TIME_RESOLUTION = 0.1s intervals) belong to each user
- This is converted to seconds: time_controlled = control_points * TIME_RESOLUTION

There's a consideration:

The current implementation is correct for measuring "time in control" - it accurately tracks how long each user's frequency setting was active. 