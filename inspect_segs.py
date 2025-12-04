import pandas as pd
df = pd.read_csv("segment_strategy_with_global_label.csv")
print(df.columns.tolist())
for seg in [11,12]:
    sel = df[df['segment_id']==seg].sort_values(['user_id'])
    print("\nSEG",seg,"rows:", len(sel))
    print(sel[['user_id','segment_id','seg_start','seg_end','num_actions','has_events',
               'structured_exploration_pred','iterative_finetuning_pred','systematic_parameter_sweep_pred']].to_string(index=False))