"""
으뜸 딥러닝 — 14장 05절
에이전트 추적 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_trace(trace):
    """Visualize the ReAct trace as a step diagram."""
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(trace)))
    colors = {"Thought": "#4A90D9", "Action": "#E8913A",
              "Observation": "#5CB85C", "Answer": "#888888"}

    y = len(trace)
    for entry in trace:
        # Determine step type
        if entry.startswith("---"):
            y -= 1
            continue
        for key, color in colors.items():
            if key in entry:
                text = entry[:80] + ("..." if len(entry) > 80
                                     else "")
                ax.barh(y, 1, color=color, alpha=0.3,
                        height=0.6)
                ax.text(0.05, y, text, va="center",
                        fontsize=8, family="monospace")
                y -= 1
                break

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    handles = [mpatches.Patch(color=c, alpha=0.4, label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8)
    plt.title("ReAct Agent Trace", fontsize=11)
    plt.tight_layout()
    plt.show()

# Visualize the trace from earlier run
visualize_trace(trace)
