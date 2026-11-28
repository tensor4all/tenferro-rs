import re,sys,statistics as st,collections
def parse(f,keyre):
    D=collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(f):
        m=re.match(r'\[(.+?) r(\d+)\] (.*)',l)
        if not m: continue
        cfg,r,rest=m.groups()
        if rest.startswith('T2742'):
            kv=dict(re.findall(r'(\w+) ([\d.]+)',rest))
            key=('run'+kv['run']) if 'run' in kv and 'positions' not in kv else ('round'+kv['round'] if 'round' in kv else 'end')
            for k,v in kv.items(): D[(cfg,key)][k].append(float(v))
        elif 'Maximum resident' in rest:
            D[(cfg,'rss')]['kb'].append(float(rest.split()[-1]))
    return D
for f in ['n30m14.txt','repro26.txt']:
    D=parse(f,None); print('#',f)
    for (cfg,key),d in D.items():
        print(cfg,key,'n=%d'%len(next(iter(d.values()))),{k:round(st.median(v),3) for k,v in d.items() if k in('seconds','highwater_over_base_gib','held_over_base_gib','entries','retained_gib','evictions','misses','kb','after_reset_gib','after_clear_gib')}, 'sec range',(min(d['seconds']),max(d['seconds'])) if 'seconds' in d else '')
