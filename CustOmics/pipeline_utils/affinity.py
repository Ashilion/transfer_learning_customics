import os

def _parse_cpu_list(s):
    out = set()
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out

def physical_core_groups(cpus, ht_offset=128):
    """Sibling HT de i = i + ht_offset (et inversement)."""
    cpus, groups, seen = set(cpus), [], set()
    for cpu in sorted(cpus):
        if cpu in seen:
            continue
        phys = cpu % ht_offset
        grp = sorted({phys, phys + ht_offset} & cpus)
        groups.append(grp)
        seen.update(grp)
    return groups

def split_cpus(cpus, n_workers):
    """Découpe `cpus` en n_workers paquets disjoints, par cœurs physiques entiers si possible."""
    units = physical_core_groups(cpus)
    if len(units) < n_workers:            # pas assez de cœurs physiques -> découpe logique
        units = [[c] for c in sorted(cpus)]
        if len(units) < n_workers:
            raise ValueError(f"{n_workers} workers pour seulement {len(units)} CPU logiques")
    q, r = divmod(len(units), n_workers)
    chunks, start = [], 0
    for i in range(n_workers):
        size = q + (1 if i < r else 0)
        chunks.append(sorted(c for u in units[start:start + size] for c in u))
        start += size
    return chunks


def split_cpus_ht_pairs(cpus, n_workers):
    """2 workers par cœur physique : chacun prend un thread HT distinct."""
    if n_workers % 2:
        raise ValueError("--multiproc doit être pair pour partager les cœurs par paires")
    groups = [g for g in physical_core_groups(cpus) if len(g) == 2]
    n_pairs = n_workers // 2
    if len(groups) < n_pairs:
        raise ValueError(f"{n_pairs} paires demandées pour {len(groups)} cœurs physiques complets")

    q, r = divmod(len(groups), n_pairs)
    chunks, start = [], 0
    for i in range(n_pairs):
        size = q + (1 if i < r else 0)
        block = groups[start:start + size]
        chunks.append([g[0] for g in block])   # threads "i"
        chunks.append([g[1] for g in block])   # threads "i + 128"
        start += size
    return chunks