import json, os
log_files = []
for root, dirs, files in os.walk('results/logs'):
    for f in files:
        if f == 'experiment_log.json' and 'tvflids' in root and 'label_flip' in root:
            log_files.append(os.path.join(root, f))

if not log_files:
    print('No TV-FLIDS logs found. Run the smoke comparison first.')
else:
    data = json.load(open(log_files[0]))
    rounds = data.get('rounds', [])
    alphas = [r.get('adaptive_alpha') for r in rounds if r.get('adaptive_alpha')]
    if not alphas:
        print('FAIL: adaptive_alpha not logged — check strategy.py log block')
    elif max(alphas) - min(alphas) < 1e-4:
        print(f'WARN: alpha range={max(alphas)-min(alphas):.6f} — weights barely moving')
    else:
        print(f'PASS: alpha range={max(alphas)-min(alphas):.4f} — meta-gradient active')
        print(f'  alpha: {alphas[0]:.4f} -> {alphas[-1]:.4f}')
