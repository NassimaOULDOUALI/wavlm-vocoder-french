#!/usr/bin/env python3
"""
Script d'analyse AMÉLIORÉ des résultats d'ablation weighted_last_n
Version robuste avec debug complet

Usage:
    python analyze_ablation_IMPROVED.py --output_dir outputs_ablation --output_csv results.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

METRICS = [
    "loss_g",
    "loss_d",
    "l1",
    "mel",
    "loss_stft",
    "loss_fm_mpd",
    "loss_fm_msd",
    "loss_gen_mpd",
    "loss_gen_msd",
]


def load_checkpoint(ckpt_path):
    """Charge un checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt
    except Exception as e:
        print(f"  ❌ Erreur chargement {ckpt_path}: {e}")
        return None


def extract_layer_weights(ckpt, n_layers, debug=True):
    """
    Extrait les poids de fusion - VERSION ROBUSTE
    """
    try:
        # Essayer différents endroits
        locations = [
            ("generator_state_dict", ckpt.get("generator_state_dict", {})),
            ("model", ckpt.get("model", {})),
            ("generator", ckpt.get("generator", {})),
            ("root", ckpt),
        ]

        for loc_name, state in locations:
            if not isinstance(state, dict):
                continue

            if debug:
                # Chercher toutes les clés contenant "layer" ou "weight"
                relevant_keys = [
                    k for k in state.keys() if "layer" in k.lower() or "weight" in k.lower()
                ]
                if relevant_keys and debug:
                    print(f"  🔍 [{loc_name}] Clés pertinentes: {relevant_keys[:3]}")

            # Chercher layer_weights
            for key in state.keys():
                if "layer_weights" in key.lower():
                    if debug:
                        print(f"  ✅ Trouvé '{key}' dans {loc_name}")

                    logits = state[key]

                    # Vérifier la taille
                    if isinstance(logits, torch.Tensor):
                        if logits.numel() != n_layers:
                            print(f"  ⚠️  Taille incorrecte: {logits.shape} (attendu {n_layers})")
                            continue

                        weights = F.softmax(logits.flatten(), dim=0).cpu().numpy()

                        if debug:
                            print(
                                f"  📊 Poids extraits: shape={weights.shape}, sum={weights.sum():.4f}"
                            )
                            print(f"      Top 3: {weights[:min(3, len(weights))]}")

                        return weights

        if debug:
            print("  ⚠️  Aucun layer_weights trouvé dans aucun emplacement")
        return None

    except Exception as e:
        print(f"  ❌ Erreur extraction: {e}")
        import traceback

        traceback.print_exc()
        return None


def parse_training_logs(log_file, debug=True):
    """
    Parse les logs - VERSION ROBUSTE
    Gère plusieurs formats de logs
    """
    metrics_by_step = defaultdict(dict)

    if debug:
        print(f"  📖 Parse log: {log_file}")

    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if debug:
            print(f"      {len(lines)} lignes lues")

        parsed_count = 0

        for line in lines:
            # Pattern: step=XXXX | loss_g=YY.YY | ...
            if "step=" not in line:
                continue

            # Extraire step
            step_match = re.search(r"step=(\d+)", line)
            if not step_match:
                continue

            step = int(step_match.group(1))

            # Extraire toutes les métriques key=value
            # Pattern robuste: capture key=value même avec plusieurs =
            metric_pattern = r"([a-z_]+)=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
            matches = re.findall(metric_pattern, line)

            for key, val in matches:
                if key == "step":
                    continue
                try:
                    metrics_by_step[step][key] = float(val)
                except (ValueError, TypeError):
                    pass

            parsed_count += 1

        if debug:
            print(f"      ✅ {parsed_count} steps parsés")
            if metrics_by_step:
                last_step = max(metrics_by_step.keys())
                print(f"      Dernier step: {last_step}")
                print(f"      Métriques: {list(metrics_by_step[last_step].keys())[:5]}")

        return dict(metrics_by_step)

    except Exception as e:
        print(f"  ❌ Erreur parsing logs: {e}")
        return {}


def analyze_n_layers_experiment(output_dir, n_values=None, debug=True):
    """
    if n_values is None:
        n_values = range(1, 13)
    Analyse l'ablation - VERSION COMPLÈTE
    """
    results = []

    for n in n_values:
        print(f"\n{'='*80}")
        print(f"📊 ANALYSE N={n}")
        print(f"{'='*80}")

        # Chercher le répertoire
        pattern = f"N{n}_*"
        matches = list(Path(output_dir).glob(pattern))

        if not matches:
            print(f"⚠️  Aucun répertoire trouvé pour pattern '{pattern}'")
            continue

        exp_dir = matches[0]
        print(f"📁 Répertoire: {exp_dir.name}")

        # Charger checkpoint
        ckpt_dir = exp_dir / "checkpoints"

        # Essayer différents checkpoints
        checkpoint_candidates = [
            ckpt_dir / "checkpoint_best.pt",
            *sorted(ckpt_dir.glob("checkpoint_step*.pt"), reverse=True),
            *sorted(ckpt_dir.glob("checkpoint_*.pt"), reverse=True),
        ]

        ckpt = None
        ckpt_path = None
        for candidate in checkpoint_candidates:
            if candidate.exists():
                print(f"🔄 Essai: {candidate.name}")
                ckpt = load_checkpoint(candidate)
                if ckpt is not None:
                    ckpt_path = candidate
                    print(f"✅ Chargé: {candidate.name}")
                    break

        if ckpt is None:
            print("❌ Aucun checkpoint valide")
            continue

        # Extraire poids
        weights = extract_layer_weights(ckpt, n, debug=debug)

        # Parser logs
        # Chercher tous les logs et prendre le plus gros (plus récent)
        log_candidates = list(Path(output_dir).parent.glob(f"slurm_*_{n}.err"))
        log_files = sorted(log_candidates, key=lambda p: p.stat().st_size, reverse=True)

        if not log_files:
            # Essayer aussi dans le répertoire courant
            log_files = list(Path(".").glob(f"slurm_*_{n}.err"))

        final_metrics = {}
        if log_files:
            print(f"📋 Log file: {log_files[0].name}")
            metrics_by_step = parse_training_logs(log_files[0], debug=debug)

            if metrics_by_step:
                last_step = max(metrics_by_step.keys())
                final_metrics = metrics_by_step[last_step]
                print(f"📊 Métriques finales (step {last_step}):")
                for k, v in list(final_metrics.items())[:5]:
                    print(f"    {k}: {v:.4f}")
        else:
            print("⚠️  Aucun fichier log trouvé")

        # Compiler résultats
        result = {
            "N": n,
            "checkpoint": ckpt_path.name if ckpt_path else "unknown",
            "step": ckpt.get("step", -1),
            "epoch": ckpt.get("epoch", -1),
        }

        # Ajouter métriques
        for metric in METRICS:
            # Essayer différentes variantes de noms
            value = final_metrics.get(metric)
            if value is None and metric.startswith("loss_"):
                # Essayer sans le préfixe loss_
                short_name = metric.replace("loss_", "")
                value = final_metrics.get(short_name)

            result[metric] = value if value is not None else np.nan

        # Ajouter poids
        if weights is not None:
            for i, w in enumerate(weights):
                result[f"weight_layer_{i}"] = w

            result["weight_entropy"] = -np.sum(weights * np.log(weights + 1e-10))
            result["weight_max"] = np.max(weights)
            result["weight_argmax"] = int(np.argmax(weights))
            result["weight_std"] = np.std(weights)

            print(f"✅ Poids extraits: {len(weights)} valeurs")
            print(f"   Argmax: layer {result['weight_argmax']} ({result['weight_max']:.3f})")
        else:
            print("⚠️  Pas de poids extraits")

        results.append(result)

    return pd.DataFrame(results)


def plot_ablation_curves(df, output_dir):
    """Génère les figures"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")

    # Figure 1: Losses
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Utiliser 'mel' au lieu de 'loss_mel'
    mel_col = "mel" if "mel" in df.columns else "loss_mel"
    l1_col = "l1" if "l1" in df.columns else "loss_l1"

    if mel_col in df.columns and not df[mel_col].isna().all():
        ax1.set_xlabel("N (nombre de couches)")
        ax1.set_ylabel("Mel Loss", color="tab:blue")
        ax1.plot(df["N"], df[mel_col], "o-", color="tab:blue", linewidth=2, markersize=8)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        if l1_col in df.columns and not df[l1_col].isna().all():
            ax2 = ax1.twinx()
            ax2.set_ylabel("L1 Loss", color="tab:orange")
            ax2.plot(df["N"], df[l1_col], "s-", color="tab:orange", linewidth=2, markersize=8)
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("Ablation: Pertes de reconstruction vs N")
        fig.tight_layout()
        plt.savefig(output_dir / "ablation_losses.pdf", dpi=300, bbox_inches="tight")
        print("✅ Figure: ablation_losses.pdf")
        plt.close()

    # Figure 2: Poids (si disponibles)
    weight_cols = [c for c in df.columns if c.startswith("weight_layer_")]

    if weight_cols:
        fig, ax = plt.subplots(figsize=(12, 6))

        for _idx, row in df.iterrows():
            n = int(row["N"])
            weights = []
            layers = []

            for i in range(n):
                col = f"weight_layer_{i}"
                if col in row and not pd.isna(row[col]):
                    weights.append(row[col])
                    layers.append(13 - n + i)  # Mapping vers couche WavLM

            if weights:
                ax.plot(layers, weights, "o-", label=f"N={n}", linewidth=2, markersize=6)

        ax.set_xlabel("Couche WavLM (0=emb, 12=dernière)")
        ax.set_ylabel("Poids softmax")
        ax.set_title("Distribution des poids appris")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(13))
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_weights.pdf", dpi=300, bbox_inches="tight")
        print("✅ Figure: ablation_weights.pdf")
        plt.close()


def generate_latex_table(df, output_file="table_ablation.tex"):
    """Génère tableau LaTeX"""

    # Trouver les bonnes colonnes
    mel_col = "mel" if "mel" in df.columns else "loss_mel"
    l1_col = "l1" if "l1" in df.columns else "loss_l1"

    cols = ["N"]
    if mel_col in df.columns:
        cols.append(mel_col)
    if l1_col in df.columns:
        cols.append(l1_col)
    if "loss_g" in df.columns:
        cols.append("loss_g")
    if "loss_d" in df.columns:
        cols.append("loss_d")

    table_df = df[cols].copy()

    # Formater
    for col in cols[1:]:  # Skip N
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "-")

    latex = table_df.to_latex(index=False, escape=False)

    with open(output_file, "w") as f:
        f.write(latex)

    print(f"✅ Tableau: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs_ablation")
    parser.add_argument("--output_csv", type=str, default="results.csv")
    parser.add_argument("--n_min", type=int, default=1)
    parser.add_argument("--n_max", type=int, default=12)
    parser.add_argument("--no-debug", action="store_true", help="Désactiver le mode debug")

    args = parser.parse_args()

    debug = not args.no_debug

    print("=" * 80)
    print("ANALYSE ABLATION weighted_last_n - VERSION AMÉLIORÉE")
    print("=" * 80)

    df = analyze_n_layers_experiment(
        args.output_dir, n_values=range(args.n_min, args.n_max + 1), debug=debug
    )

    if df.empty:
        print("\n❌ Aucun résultat")
        return

    # Sauvegarder
    df.to_csv(args.output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"✅ CSV sauvegardé: {args.output_csv}")

    # Afficher résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print("=" * 80)

    display_cols = ["N", "epoch", "step"]
    for col in ["mel", "loss_mel", "l1", "loss_l1", "loss_g", "loss_d", "weight_argmax"]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    # Générer figures
    plot_ablation_curves(df, "figures_ablation")

    # Générer tableau
    generate_latex_table(df, "table_ablation.tex")

    print(f"\n{'='*80}")
    print("✅ TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()
