"""
Rebuild pages.csv and edges.csv from categories.csv.
Includes one-hop neighbors of category pages and removes dangling node references.
"""
import pandas as pd

DATA = "/Users/noertoftpm/Classes/MATH171/WIM/data"

# Load desired categories, preserving order from CSV
cats_df = pd.read_csv(f"{DATA}/categories.csv")
desired_ordered = list(cats_df["category"].str.strip())
desired = set(desired_ordered)
cat_order = {cat: i for i, cat in enumerate(desired_ordered)}
print(f"Desired categories: {len(desired)}")

# Collect seed page IDs; assign each page to its first-seen category
seed_ids = set()
page_category = {}   # page_id -> category name (first assignment wins)
found_cats = set()
with open(f"{DATA}/wiki-topcats/wiki-topcats-categories.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        semi = line.index(";")
        cat = line[:semi].strip()
        if cat in desired:
            found_cats.add(cat)
            ids = [int(x) for x in line[semi + 1:].split()]
            for pid in ids:
                seed_ids.add(pid)
                if pid not in page_category:
                    page_category[pid] = cat

print(f"Found {len(found_cats)}/{len(desired)} categories, {len(seed_ids)} seed pages")
missing = desired - found_cats
if missing:
    print("Missing categories:", missing)

# Load all edges, filter to induced subgraph over category nodes
print("Loading all edges...")
all_edges = pd.read_csv(
    f"{DATA}/wiki-topcats/wiki-topcats.txt",
    sep=" ", header=None, names=["src", "dst"]
)
print(f"Total edges: {len(all_edges)}")

extended_ids = seed_ids
print(f"Node set: {len(extended_ids)}")

# Induced subgraph: edges where both endpoints are category nodes
mask = all_edges["src"].isin(extended_ids) & all_edges["dst"].isin(extended_ids)
filtered_edges_df = all_edges[mask].copy()
print(f"Induced subgraph edges: {len(filtered_edges_df)}")

# Load page names for nodes in the extended set
print("Loading page names...")
page_names = {}
with open(f"{DATA}/wiki-topcats/wiki-topcats-page-names .txt") as f:
    for line in f:
        line = line.strip()
        if not line or " " not in line:
            continue
        space = line.index(" ")
        pid = int(line[:space])
        if pid in extended_ids:
            page_names[pid] = line[space + 1:]

# Final nodes: those in edges AND have a known page name
edge_node_ids = set(filtered_edges_df["src"]) | set(filtered_edges_df["dst"])
valid_ids = edge_node_ids & set(page_names.keys())

# Sort alphabetically by category name, then by page title within category
final_ids = sorted(valid_ids, key=lambda pid: (page_category.get(pid, ""), page_names[pid]))
print(f"Final node count: {len(final_ids)}")

# Re-index: old_id -> new_id (0-based)
id_map = {old: new for new, old in enumerate(final_ids)}
final_id_set = set(final_ids)

# Write pages.csv
pages_df = pd.DataFrame({
    "id": range(len(final_ids)),
    "title": [page_names[i] for i in final_ids],
    "category": [page_category.get(i, "") for i in final_ids],
})
pages_df.to_csv(f"{DATA}/pages.csv", index=False)
print(f"Wrote pages.csv: {len(pages_df)} rows")

# Write edges.csv — remap IDs and drop any edges with nodes missing from final set
out_edges = filtered_edges_df[
    filtered_edges_df["src"].isin(final_id_set) & filtered_edges_df["dst"].isin(final_id_set)
].copy()
out_edges["src"] = out_edges["src"].map(id_map)
out_edges["dst"] = out_edges["dst"].map(id_map)
out_edges.to_csv(f"{DATA}/edges.csv", index=False)
print(f"Wrote edges.csv: {len(out_edges)} rows")