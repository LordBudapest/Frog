import os, glob, json, argparse
import numpy as np

def load_jsonl(glob_pattern):
    rows = []
    for path in glob.glob(glob_pattern):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows
def summarize_pcqm(rows, ddof = 1):
    by_mode = {}
    k_val = None
    for r in rows:
        if r.get('dataset') != 'PCQM-Contact':
            continue
        m = r['mode']
        by_mode.setdefault(m, {'hits': [], 'mrr': []})
        by_mode[m]['hits'].append(float(r['hits_at_k']))
        by_mode[m]['mrr'].append(float(r['mrr']))
        if k_val is None and 'k' in r:
            k_val = int(r['k'])
    if k_val is None:
        k_val = 20
    print('PCQM-Contact aggregate (mean ± std over seeds):')
    for m in sorted(by_mode.keys()):
        h = np.array(by_mode[m]['hits'], dtype=float)
        s = np.array(by_mode[m]['mrr'], dtype=float)
        h_mu = float(h.mean()) if h.size else 0.0
        h_sd = float(h.std(ddof=ddof)) if h.size > 1 else 0.0
        s_mu = float(s.mean()) if s.size else 0.0
        s_sd = float(s.std(ddof=ddof)) if s.size > 1 else 0.0
        print(f'{m}: Hits@{k_val} = {h_mu:.4f} ± {h_sd:.4f}; MRR = {s_mu:.4f} ± {s_sd:.4f}')

def summarize_peptides(rows, ddof = 1):
    by_mode = {}
    for r in rows:
        if r.get('dataset') != 'peptides-func':
            continue
        m = r['mode']
        by_mode.setdefault(m, [])
        by_mode[m].append(float(r['ap']))
    print('peptides-func aggregate (mean ± std over seeds):')
    for m in sorted(by_mode.keys()):
        a = np.array(by_mode[m], dtype=float)
        mu = float(a.mean()) if a.size else 0.0
        sd = float(a.std(ddof=ddof)) if a.size > 1 else 0.0
        print(f"{m}: AP = {mu:.4f} ± {sd:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pcqm_glob', default='results/pcqm_contact_seed*.jsonl')
    p.add_argument('--peptides_glob', default='results/peptides_func_seed*.jsonl')
    p.add_argument('--ddof', type=int, default=1, help='std degrees of freedom (1 for sample std)')
    args = p.parse_args()
    rows = []
    rows += load_jsonl(args.pcqm_glob)
    rows += load_jsonl(args.peptides_glob)

    pcqm_rows = [r for r in rows if r.get('dataset') == 'PCQM-Contact']
    pep_rows = [r for r in rows if r.get('dataset') == 'peptides-func']

    if pcqm_rows:
        summarize_pcqm(pcqm_rows, ddof=args.ddof)
        print()
    else:
        print('No PCQM-Contact rows found for pattern:', args.pcqm_glob)
    if pep_rows:
        summarize_peptides(pep_rows, ddof=args.ddof)
    else:
        print('No peptides-func rows found for pattern:', args.peptides_glob)

if __name__ == '__main__':
    main()