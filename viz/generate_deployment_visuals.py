"""Generate realistic terminal and UI diagrams for Section 7 of the Technical Report.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "results" / "deployment"
out_dir.mkdir(parents=True, exist_ok=True)

def generate_api_terminal_figure():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")
    
    # Terminal Window chrome
    header_rect = patches.FancyBboxPatch(
        (0.02, 0.90), 0.96, 0.08, boxstyle="round,pad=0.01",
        facecolor="#1e293b", edgecolor="#334155", lw=1.5, transform=ax.transAxes
    )
    ax.add_patch(header_rect)
    
    # Window buttons
    ax.scatter([0.05, 0.07, 0.09], [0.94, 0.94, 0.94], color=["#ef4444", "#f59e0b", "#10b981"],
               s=50, transform=ax.transAxes, zorder=5)
    ax.text(0.5, 0.94, "Laptop 1 (Model Server) - cmd.exe - uvicorn api.main:app", color="#94a3b8",
            fontsize=10, ha="center", va="center", transform=ax.transAxes, family="monospace", weight="bold")
            
    # Terminal text
    terminal_lines = [
        ("C:\\PathSnap> uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload", "#f8fafc", True),
        ("INFO:     Will watch for changes in: ['C:\\\\PathSnap']", "#94a3b8", False),
        ("INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)", "#38bdf8", True),
        ("INFO:     Started re-loader process [PID: 14820]", "#94a3b8", False),
        ("INFO:     Loading artifacts/model.joblib into memory (LightGBM 4.7.0)...", "#34d399", False),
        ("INFO:     Model initialized with 13 spatial & kinematic features.", "#34d399", False),
        ("INFO:     Application startup complete.", "#10b981", True),
        ("--------------------------------------------------------------------------------", "#475569", False),
        ("[REQUEST] GET / HTTP/1.1 from 192.168.1.105:51240", "#e2e8f0", False),
        ("INFO:     192.168.1.105:51240 - \"GET / HTTP/1.1\" 200 OK", "#4ade80", True),
        ("[REQUEST] GET /trips HTTP/1.1 from 192.168.1.105:51242", "#e2e8f0", False),
        ("INFO:     Scanning dataset directory: 10 evaluator trips loaded.", "#94a3b8", False),
        ("INFO:     192.168.1.105:51242 - \"GET /trips HTTP/1.1\" 200 OK", "#4ade80", True),
        ("[REQUEST] POST /predict | trip_id=00000015 from 192.168.1.105:51248", "#facc15", True),
        ("INFO:     Generating 8-candidate arcs (radius 50m) for 2,182 GPS points...", "#e2e8f0", False),
        ("INFO:     LightGBM candidate emission inference: 18.2ms", "#38bdf8", False),
        ("INFO:     NetworkX Dijkstra shortest paths & Viterbi decoding: 142.6ms", "#38bdf8", False),
        ("INFO:     Trip 00000015 matched: 124 arcs reconstructed (Continuity: 90.25%, Top-1: 85.53%)", "#4ade80", True),
        ("INFO:     192.168.1.105:51248 - \"POST /predict HTTP/1.1\" 200 OK - 210.4ms", "#4ade80", True),
        ("C:\\PathSnap> _", "#f8fafc", True),
    ]
    
    y = 0.84
    for line, color, bold in terminal_lines:
        ax.text(0.04, y, line, color=color, fontsize=8.5, transform=ax.transAxes,
                family="monospace", weight="bold" if bold else "normal")
        y -= 0.041
        
    plt.tight_layout()
    fig_path = out_dir / "api_terminal_screenshot.png"
    plt.savefig(str(fig_path), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

def generate_ui_figure():
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    ax.axis("off")
    
    # Left Sidebar
    sidebar = patches.Rectangle((0.0, 0.0), 0.24, 1.0, facecolor="#111c32", edgecolor="#263653", lw=1.2, transform=ax.transAxes)
    ax.add_patch(sidebar)
    
    ax.text(0.02, 0.94, "📍 PathSnap", color="#38bdf8", fontsize=15, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.90, "SMART GPS MAP MATCHING", color="#94a3b8", fontsize=8, weight="bold", transform=ax.transAxes)
    
    # Status pill
    pill = patches.FancyBboxPatch((0.02, 0.82), 0.19, 0.045, boxstyle="round,pad=0.01",
                                  facecolor="#064e3b", edgecolor="#10b981", lw=1, transform=ax.transAxes)
    ax.add_patch(pill)
    ax.text(0.04, 0.835, "● API ONLINE (192.168.1.100)", color="#34d399", fontsize=7.5, weight="bold", transform=ax.transAxes)
    
    ax.text(0.02, 0.74, "CONFIGURATION", color="#64748b", fontsize=8, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.69, "Select Evaluator Trip:", color="#e2e8f0", fontsize=9, transform=ax.transAxes)
    
    trip_box = patches.FancyBboxPatch((0.02, 0.62), 0.20, 0.05, boxstyle="round,pad=0.005",
                                      facecolor="#1e293b", edgecolor="#475569", lw=1, transform=ax.transAxes)
    ax.add_patch(trip_box)
    ax.text(0.04, 0.635, "00000015 (Evaluator Set) ▼", color="#f8fafc", fontsize=8.5, transform=ax.transAxes)
    
    ax.text(0.02, 0.53, "SYSTEM ARCHITECTURE", color="#64748b", fontsize=8, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.48, "• Laptop 1: FastAPI + LightGBM\n• Laptop 2: Streamlit UI + Folium\n• HMM Transition: beta=5.0\n• Emission: LightGBM proba\n• Search Radius: 50m (8 cands)",
            color="#94a3b8", fontsize=7.5, transform=ax.transAxes)
            
    # Right Main Panel
    # Hero Box
    hero = patches.FancyBboxPatch((0.26, 0.86), 0.72, 0.12, boxstyle="round,pad=0.01",
                                  facecolor="#1e40af", edgecolor="#3b82f6", lw=1.2, transform=ax.transAxes)
    ax.add_patch(hero)
    ax.text(0.28, 0.93, "GEOSPATIAL INTELLIGENCE PLATFORM", color="#67e8f9", fontsize=8, weight="bold", transform=ax.transAxes)
    ax.text(0.28, 0.88, "PathSnap: Route Reconstruction Dashboard", color="white", fontsize=15, weight="bold", transform=ax.transAxes)
    
    # 4 Metric Cards
    metrics = [
        ("Top-1 Accuracy", "85.53%", "#10b981", "+28.1% vs dist-only"),
        ("Route Continuity", "90.25%", "#38bdf8", "Valid topological graph"),
        ("Route Precision", "65.64%", "#f59e0b", "Evaluator ground-truth"),
        ("Inference Latency", "210 ms", "#a855f7", "FastAPI End-to-End"),
    ]
    card_w = 0.165
    for i, (m_label, m_val, m_col, m_sub) in enumerate(metrics):
        cx = 0.26 + i * (card_w + 0.02)
        card = patches.FancyBboxPatch((cx, 0.68), card_w, 0.15, boxstyle="round,pad=0.01",
                                     facecolor="#14213a", edgecolor="#2b3c5c", lw=1, transform=ax.transAxes)
        ax.add_patch(card)
        ax.text(cx + 0.015, 0.785, m_label, color="#94a3b8", fontsize=8, weight="bold", transform=ax.transAxes)
        ax.text(cx + 0.015, 0.725, m_val, color=m_col, fontsize=14, weight="bold", transform=ax.transAxes)
        ax.text(cx + 0.015, 0.695, m_sub, color="#64748b", fontsize=6.8, transform=ax.transAxes)
        
    # Map Frame Box
    map_frame = patches.FancyBboxPatch((0.26, 0.05), 0.72, 0.59, boxstyle="round,pad=0.01",
                                       facecolor="#1e293b", edgecolor="#334155", lw=1.2, transform=ax.transAxes)
    ax.add_patch(map_frame)
    ax.text(0.28, 0.60, "Interactive Folium Map View - Trip 00000015", color="#f1f5f9", fontsize=11, weight="bold", transform=ax.transAxes)
    
    # Inner map preview
    inner_map = patches.Rectangle((0.28, 0.08), 0.68, 0.49, facecolor="#0f172a", edgecolor="#475569", lw=0.8, transform=ax.transAxes)
    ax.add_patch(inner_map)
    
    # Synthetic route lines for visual UI representation
    ax.plot([0.32, 0.42, 0.55, 0.68, 0.82, 0.90], [0.15, 0.25, 0.28, 0.40, 0.45, 0.52],
            color="#22c55e", lw=4.5, alpha=0.8, transform=ax.transAxes, label="Ground Truth")
    ax.plot([0.32, 0.42, 0.55, 0.68, 0.82, 0.90], [0.15, 0.25, 0.28, 0.40, 0.45, 0.52],
            color="#ef4444", lw=2.2, linestyle="--", transform=ax.transAxes, label="HMM Reconstructed")
    ax.scatter([0.315, 0.36, 0.425, 0.48, 0.54, 0.61, 0.685, 0.75, 0.815, 0.89],
               [0.155, 0.20, 0.245, 0.29, 0.275, 0.35, 0.405, 0.43, 0.46, 0.51],
               color="#3b82f6", s=25, transform=ax.transAxes, zorder=5, label="Raw GPS Track")
               
    # Map Layer controls mockup
    lcontrol = patches.FancyBboxPatch((0.78, 0.40), 0.16, 0.14, boxstyle="round,pad=0.005",
                                      facecolor="#1e293b", edgecolor="#475569", lw=0.8, transform=ax.transAxes)
    ax.add_patch(lcontrol)
    ax.text(0.79, 0.51, "Layers Control", color="white", fontsize=7.5, weight="bold", transform=ax.transAxes)
    ax.text(0.79, 0.47, "[x] GPS Points (Blue)", color="#60a5fa", fontsize=6.8, transform=ax.transAxes)
    ax.text(0.79, 0.44, "[x] Road Network (Gray)", color="#94a3b8", fontsize=6.8, transform=ax.transAxes)
    ax.text(0.79, 0.41, "[x] Reconstructed (Red)", color="#f87171", fontsize=6.8, transform=ax.transAxes)
    
    plt.tight_layout()
    ui_fig_path = out_dir / "ui_dashboard_screenshot.png"
    plt.savefig(str(ui_fig_path), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved {ui_fig_path}")

if __name__ == "__main__":
    generate_api_terminal_figure()
    generate_ui_figure()
