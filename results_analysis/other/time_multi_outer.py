import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import pandas as pd

rows = []

with open("results/time_multi_outer.txt") as f:
    for line in f:

        cols = line.split()

        if len(cols) < 6:
            continue

        jobid = cols[0]
        name = cols[1]
        state = cols[2]

        # ne garder que le job principal
        if name != "custcox_n+":
            continue

        m = re.search(r"_(\d+)$", jobid)
        trial = int(m.group(1))

        start = cols[3]
        end = cols[4]

        rows.append(
            {
                "trial": trial,
                "state": state,
                "start": start,
                "end": end,
            }
        )

df = pd.DataFrame(rows).sort_values("trial")

df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])

df["duration"] = (df["end"] - df["start"]).dt.total_seconds() / 86400

colors = {
    "COMPLETED": "tab:green",
    "FAILED": "tab:red",
    "TIMEOUT": "tab:orange",
    "CANCELLED": "tab:gray",
}

fig, ax = plt.subplots(figsize=(14, 10))

for i, row in enumerate(df.itertuples()):
    ax.barh(
        y=i,
        width=row.duration,
        left=mdates.date2num(row.start),
        height=0.3,
        color=colors.get(row.state, "tab:blue"),
        edgecolor="black",
        linewidth=0.3,
    )

ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["trial"])

ax.xaxis_date()
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d\n%H:%M")
)

ax.set_xlabel("Time")
ax.set_ylabel("Trial")
ax.set_title("Trial Timeline")

plt.tight_layout()
plt.show()