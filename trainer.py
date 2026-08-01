"""
trainer.py  —  Orchestrator for CSI-DRL Engine
===============================================

Loads data → discovers causal graphs → trains DRL agent → builds JSON.
Uses parallel processing for speed.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from csi_drl import compute_universe_csi_drl
from push_results import upload_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_action(z_score: float, all_z_scores: List[float] = None) -> str:
    """
    Determine action based on z-score percentile.
    Uses percentile-based thresholds for proper BUY/SELL distribution.
    """
    if all_z_scores is None or len(all_z_scores) < 2:
        # Fallback: simple thresholds
        if z_score > 0.15:
            return "BUY"
        elif z_score > -0.15:
            return "HOLD"
        else:
            return "SELL"
    
    # Percentile-based thresholds
    p80 = np.percentile(all_z_scores, 80)
    p40 = np.percentile(all_z_scores, 40)
    p20 = np.percentile(all_z_scores, 20)
    p90 = np.percentile(all_z_scores, 90)
    p10 = np.percentile(all_z_scores, 10)
    
    if z_score > p80:
        return "STRONG BUY" if z_score > p90 else "BUY"
    elif z_score > p40:
        return "HOLD"
    elif z_score > p20:
        return "REDUCE"
    else:
        return "STRONG SELL" if z_score < p10 else "SELL"


def process_window(args: Tuple) -> Dict:
    """Process a single window for a universe in parallel."""
    window, universe_name, available, prices_df, macro_df, config_dict = args
    
    try:
        universe_prices = prices_df[available]
        result = compute_universe_csi_drl(universe_prices, macro_df, config_dict, window)
        
        return {
            "window": window,
            "universe": universe_name,
            "results": result,
            "error": None
        }
    except Exception as e:
        return {
            "window": window,
            "universe": universe_name,
            "results": {},
            "error": str(e)
        }


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Run the full CSI-DRL pipeline."""
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"✅ Loaded {len(prices_df)} days, {len(prices_df.columns)} ETFs")

    run_date = datetime.now().strftime("%Y-%m-%d")

    engine_config = {
        **config.CAUSAL,
        **config.GNN,
        **config.RL,
    }

    results_tab1 = {"run_date": run_date, "universes": {}}
    results_tab2 = {"run_date": run_date, "universes": {}}

    # ── Prepare parallel tasks ───────────────────────────────────────────────
    tasks = []
    windows = config.WINDOWS
    max_workers = max(1, int(mp.cpu_count() * 0.75))
    logger.info(f"🚀 Using {max_workers} parallel workers")

    for universe_name, tickers in config.UNIVERSES.items():
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        for window in windows:
            tasks.append((window, universe_name, available, prices_df, macro_df, engine_config))

    # ── Run parallel processing ──────────────────────────────────────────────
    logger.info(f"📋 Total tasks: {len(tasks)}")
    all_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_window, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result(timeout=1800)
                if result.get("error"):
                    logger.warning(f"   ⚠️ {result['universe']} @ {result['window']}d failed: {result['error']}")
                    continue
                key = f"{result['universe']}_{result['window']}"
                all_results[key] = result
                logger.info(f"   ✅ [{completed}/{len(tasks)}] {result['universe']} @ {result['window']}d")
            except Exception as e:
                logger.error(f"   ❌ Task failed: {e}")

    logger.info(f"✅ Completed {len(all_results)}/{len(tasks)} tasks")

    # ── Build results ──────────────────────────────────────────────────────────
    for universe_name in config.UNIVERSES.keys():
        available = [t for t in config.UNIVERSES[universe_name] if t in prices_df.columns]
        if not available:
            continue

        universe_results = {}
        for key, result in all_results.items():
            if result.get("universe") == universe_name:
                universe_results[str(result["window"])] = result

        if not universe_results:
            continue

        # Collect all z-scores for percentile-based actions
        all_z_scores_global = []
        for window, wr in universe_results.items():
            for ticker_data in wr.get("results", {}).values():
                z = safe_float(ticker_data.get("z_score", 0))
                all_z_scores_global.append(z)
        
        all_z_scores_global = np.array(all_z_scores_global) if all_z_scores_global else np.array([0])

        # ── Build Tab 1: Best window per ETF ──────────────────────────────────
        best_window_per_etf = {}
        for ticker in available:
            best_z = -999
            best_win = None
            best_data = None
            for window, wr in universe_results.items():
                ticker_data = wr.get("results", {}).get(ticker, {})
                z = safe_float(ticker_data.get("z_score", -999))
                if z > best_z:
                    best_z = z
                    best_win = window
                    best_data = ticker_data
            if best_win is not None:
                # Use the action from the computed result
                action = best_data.get("action", "HOLD")
                if action == "PENDING" or action == "HOLD":
                    action = get_action(best_z, all_z_scores_global.tolist())
                
                best_window_per_etf[ticker] = {
                    "z_score": best_z,
                    "window": int(best_win),
                    "action": action,
                    "position": safe_float(best_data.get("position", 0)),
                    "causal_links": int(safe_float(best_data.get("causal_links", 0))),
                    "action_probabilities": best_data.get("action_probabilities", [0.33, 0.33, 0.34])
                }

        if not best_window_per_etf:
            continue

        top_buys = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        top_sells = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1]
        )[:5]

        results_tab1["universes"][universe_name] = {
            "top_buys": [{"ticker": t, "z_score": z} for t, z in top_buys],
            "top_sells": [{"ticker": t, "z_score": z} for t, z in top_sells],
            "full_scores": best_window_per_etf
        }

        # ── Tab 2: Per-window breakdown ───────────────────────────────────────
        results_tab2["universes"][universe_name] = {
            "windows": {
                window: {
                    "top_buys": [
                        {"ticker": t, "z_score": z}
                        for t, z in sorted(
                            [(t, safe_float(wr.get("results", {}).get(t, {}).get("z_score", 0)))
                             for t in available],
                            key=lambda x: x[1], reverse=True
                        )[:5]
                    ],
                    "full_ranking": [
                        [
                            t,
                            safe_float(wr.get("results", {}).get(t, {}).get("z_score", 0)),
                            # FIX: Compute action for each window individually using that window's z-scores
                            get_action(
                                safe_float(wr.get("results", {}).get(t, {}).get("z_score", 0)),
                                [safe_float(wr.get("results", {}).get(x, {}).get("z_score", 0)) for x in available]
                            )
                        ]
                        for t in available
                    ]
                }
                for window, wr in universe_results.items()
            }
        }

        logger.info(f"   ✅ {universe_name}: {len(best_window_per_etf)} ETFs ranked")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")
    tab1_path = f"csi_drl_{run_date}.json"
    tab2_path = f"csi_drl_breakdown_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2, default=str)
    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2, default=str)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    # ── Upload ─────────────────────────────────────────────────────────────────
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            api = HfApi(token=token)
            for path in [tab1_path, tab2_path]:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=path,
                    repo_id=config.RESULTS_REPO,
                    token=token,
                    repo_type="dataset"
                )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
