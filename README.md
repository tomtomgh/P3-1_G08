# Notes


### Current Dominance Calculation:


- For shared parameters (like frequency), the system tracks which user "owns" each time point in user_at_time
- When a user changes the frequency, they become the owner from that moment until another user changes it
- Dominance is calculated by counting how many timeline points (at TIME_RESOLUTION = 0.1s intervals) belong to each user
- This is converted to seconds: time_controlled = control_points * TIME_RESOLUTION

There's a consideration:

The current implementation is correct for measuring "time in control" - it accurately tracks how long each user's frequency setting was active. 

### How to Test Classfifer Accuracy
1. Evaluate existing labeled data:
```
python tests/evaluate_classifier.py --labeled-dir logs/Labelled
```
2. Run synthetic test:
```
python tests/evaluate_classifier.py --synthetic
```
3. Create labels for new data:
```
python tests/ground_truth_format.py --create-template path/to/ground_truth_labels.json
```
