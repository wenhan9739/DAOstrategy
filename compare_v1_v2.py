"""Quick numeric comparison: v1 BC vs v2 BC vs training distribution."""
import json

v1 = json.load(open('stats_selfplay.json', encoding='utf-8'))
v2 = json.load(open('stats_selfplay_v2.json', encoding='utf-8'))
tr = json.load(open('stats_training.json', encoding='utf-8'))


def tv(a, b):
    n = min(len(a), len(b))
    return 0.5 * sum(abs(a[i] - b[i]) for i in range(n))


def rb_freq(counts):
    s = sum(counts)
    return [c / s if s > 0 else 0.0 for c in counts]


print('=== entropy ===')
print(f'  v1 policy entropy: {v1["policy_entropy_nats_avg"]:.3f} nats')
print(f'  v2 policy entropy: {v2["policy_entropy_nats_avg"]:.3f} nats')

print()
print('=== raise bucket freq (in all raise actions) ===')
labs = tr['raise_bucket_labels']
tr_rb = rb_freq(tr['raise_bucket_count'])
v1_rb = rb_freq(v1['raise_bucket_count'])
v2_rb = rb_freq(v2['raise_bucket_count'])
print(f'  {"bucket":<10}  train%   v1%     v2%')
for i, l in enumerate(labs):
    print(f'  {l:<10}  {tr_rb[i]*100:5.1f}%  {v1_rb[i]*100:5.1f}%  {v2_rb[i]*100:5.1f}%')
print(f'  TV(train,v1) = {tv(tr_rb, v1_rb)*100:.1f}%')
print(f'  TV(train,v2) = {tv(tr_rb, v2_rb)*100:.1f}%')

print()
print('=== overall action TV to training ===')
print(f'  v1 TV = {tv(tr["action_freq_overall"], v1["action_freq_overall"])*100:.1f}%')
print(f'  v2 TV = {tv(tr["action_freq_overall"], v2["action_freq_overall"])*100:.1f}%')

print()
print('=== per-street TV to training ===')
for i, s in enumerate(['pre', 'flop', 'turn', 'river']):
    t1 = tv(tr['action_freq_per_street'][i], v1['action_freq_per_street'][i])
    t2 = tv(tr['action_freq_per_street'][i], v2['action_freq_per_street'][i])
    print(f'  {s:<7}: v1={t1*100:.1f}%  v2={t2*100:.1f}%')

print()
print('=== pf_open (first voluntary, 0 raises) per position ===')


def agg(row):
    return row[0], sum(row[2:])


for i, p in enumerate(tr['pos_names']):
    t_f, t_r = agg(tr['pf_open_freq_per_pos'][i])
    v1_f, v1_r = agg(v1['pf_open_freq_per_pos'][i])
    v2_f, v2_r = agg(v2['pf_open_freq_per_pos'][i])
    print(f'  {p:<4}: train fold={t_f*100:4.1f}% raise={t_r*100:4.1f}% | '
          f'v1 fold={v1_f*100:4.1f}% raise={v1_r*100:4.1f}% | '
          f'v2 fold={v2_f*100:4.1f}% raise={v2_r*100:4.1f}%')

print()
print('=== hand-level ===')
for k in ['showdown_rate', 'all_in_rate', 'avg_actions_per_hand']:
    print(f'  {k:<22}  v1={v1[k]:.4f}  v2={v2[k]:.4f}')

print()
print('=== per-seat VPIP / PFR ===')
for i, p in enumerate(v1['pos_names']):
    print(f'  {p:<4}: v1 VPIP={v1["vpip_per_seat"][i]*100:5.1f}% PFR={v1["pfr_per_seat"][i]*100:5.1f}% | '
          f'v2 VPIP={v2["vpip_per_seat"][i]*100:5.1f}% PFR={v2["pfr_per_seat"][i]*100:5.1f}%')
