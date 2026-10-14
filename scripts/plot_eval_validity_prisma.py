"""PRISMA-style flow diagram for the evaluation-validity literature review (2026-10-01)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "data/processed/lit_review_eval_validity_20261001/fig_prisma_eval_validity.png"
fig, ax = plt.subplots(figsize=(10, 7.6))
ax.set_xlim(0, 10)
ax.set_ylim(1.2, 10)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="#33475b", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#33475b", lw=1.1))


box(0.3, 8.4, 5.6, 1.3,
    "Europe PMC, 6 field-restricted queries: 1,286 records\n"
    "(leakage/split 559, autocorrelation 13, time-series null 82,\n"
    "chance level 368, CV uncertainty 215, BCI evaluation 49)\n"
    "+ arXiv: 38 records   (total 1,324)")
box(0.3, 6.9, 5.6, 1.0, "After de-duplication: 1,282 records (42 duplicates)")
box(6.3, 6.9, 3.4, 1.0, "Excluded by title keyword screen\n(not methodological): 1,138", fc="#f7eeee")
box(0.3, 5.4, 5.6, 1.0, "Titles screened manually: 144")
box(6.3, 5.4, 3.4, 1.0, "Excluded (off-topic domain,\napplication papers, duplicates): 112", fc="#f7eeee")
box(0.3, 3.9, 5.6, 1.0, "Abstracts assessed: 32")
box(6.3, 3.9, 3.4, 1.0, "Excluded: 1 (no abstract available)", fc="#f7eeee")
box(0.3, 1.6, 5.6, 1.6, "Included from database search: 31\n+ classics: 19\nTotal included: 50")
box(6.3, 1.6, 3.4, 1.6,
    "Classics by targeted\nCrossref / Europe PMC lookup:\n18 queries + 3 previously\nverified -> 19 included;\n1 not verifiable (excluded)", fc="#f3f6ee")

for y1, y2 in [(8.4, 7.9), (6.9, 6.4), (5.4, 4.9), (3.9, 3.2)]:
    arrow(3.1, y1, 3.1, y2)
for y in [7.4, 5.9, 4.4]:
    arrow(5.9, y, 6.3, y)
arrow(6.3, 2.4, 5.9, 2.4)
ax.set_title("Evaluation-validity review: identification and screening\n(single reviewer, mostly abstract-level appraisal)", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(OUT)
