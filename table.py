"""

Term Frequency Table
Date: 10-20-26

"""

import textwrap
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from data_analysis import get_entity_frequency

entity_frequency = get_entity_frequency()

counters = dict(entity_frequency)
labels = list(counters.keys())

DEFAULT_N = 10
MAX_N_CAP = 50
LEGEND_WRAP = 24
CELL_MAX_CHARS = 45
COLS = ["Word", "Frequency"]

max_available = max((len(c) for c in counters.values()), default=1)
MAX_N = max(1, min(max_available, MAX_N_CAP))
DEFAULT_N = min(DEFAULT_N, MAX_N)

def wrap(text, width):
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)

def shorten(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "..."

wrapped_labels = [wrap(l, LEGEND_WRAP) for l in labels]
legend_chars = max((len(line) for wl in wrapped_labels for line in wl.split("\n")), default=10)
legend_lines = sum(wl.count("\n") + 1 for wl in wrapped_labels)

word_chars = max(
    (len(shorten(w, CELL_MAX_CHARS)) for c in counters.values() for w, _ in c.most_common(MAX_N)),
    default = 4
)

freq_chars = max(
    (len(f"{n:,}") for c in counters.values() for _, n in c.most_common(MAX_N)),
    default=4
)

word_chars = max(word_chars, len(COLS[0]))
freq_chars = max(freq_chars, len(COLS[1]))

CHAR_IN = 0.1
legend_w_in = legend_chars * CHAR_IN + 0.6
table_w_in = (word_chars + freq_chars) * CHAR_IN + 1.0
fig_w = min(max(legend_w_in + table_w_in + 0.8, 7), 16)
fig_h = min(max(0.28 * min(MAX_N, 25) + 2.0, 4.5), 10)

fig = plt.figure(figsize=(fig_w, fig_h))

margin = 0.02
legend_w = min(legend_w_in / fig_w, 0.4)
legend_h = min((0.3 * legend_lines + 0.3) / fig_h, 0.8)
table_left = margin + legend_w + 0.03
slider_h = 0.04
table_bottom = 0.14

tab_ax = fig.add_axes([margin, 0.5 - legend_h / 2, legend_w, legend_h])
selector = RadioButtons(tab_ax, wrapped_labels)
for txt in selector.labels:
    txt.set_fontsize(10)
label_lookup = dict(zip(wrapped_labels, labels))

ax = fig.add_axes([table_left, table_bottom, 1 - table_left - margin, 0.95 - table_bottom])
ax.axis("off")

slider_ax = fig.add_axes([table_left + 0.08, 0.04, 0.6 * (1 - table_left - margin), slider_h])
slider = Slider(slider_ax, "Top N", 1, MAX_N, valinit=DEFAULT_N, valstep=1, valfmt="%d")

current_table = None

def draw_table(*_):
    global current_table

    if current_table is not None:
        current_table.remove()
        current_table = None

    label = label_lookup[selector.value_selected]
    n = int(slider.val)
    rows = [
        [shorten(w, CELL_MAX_CHARS), f"{c:,}"] for w, c in counters[label].most_common(n)
    ]

    if not rows: rows = [["(no data)", ""]]

    wc = max(max(len(r[0]) for r in rows), len(COLS[0])) + 2
    fc = max(max(len(r[1]) for r in rows), len(COLS[1])) + 2

    col_widths = [wc / (wc + fc), fc / (wc + fc)]

    current_table = ax.table(
        cellText=rows,
        colLabels=COLS,
        colWidths=col_widths,
        cellLoc="center",
        bbox=[0,0,1,1]
    )

    ax_h_pt = ax.bbox.height * 72 / fig.dpi
    ax_w_pt = ax.bbox.width * 72 / fig.dpi

    by_height = ax_h_pt / (len(rows) + 1) * 0.55
    by_width = ax_w_pt * col_widths[0] / (wc * 0.6)

    fontsize = max(6, min(12, by_height, by_width))

    current_table.auto_set_font_size(False)
    current_table.set_fontsize(fontsize)

    for col in range(len(COLS)):
        current_table[0, col].set_facecolor("#4472C4")
        current_table[0, col].set_text_props(color="white", weight="bold")

    ax.set_title(f'{label}: top {n}', fontsize=12, pad=8)
    fig.canvas.draw_idle()

selector.on_clicked(draw_table)
slider.on_changed(draw_table)
fig.canvas.mpl_connect("resize_event", draw_table)

draw_table()
plt.show()