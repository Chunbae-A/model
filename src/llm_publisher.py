from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def _find_latest_run(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        return None
    runs.sort()
    return runs[-1]


def _load_json_safe(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_model_txts(models_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not models_dir.exists():
        return out
    for f in models_dir.iterdir():
        if f.suffix.lower() == ".txt":
            lines = f.read_text(encoding="utf-8").splitlines()
            records = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    body = line.lstrip("- ")
                    try:
                        records.append(json.loads(body))
                    except Exception:
                        records.append({"text": body})
            out[f.stem] = records
    return out


def _read_csv_sample(path: Path, n: int | None = 10) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        if n is None:
            return df.to_dict(orient="records")
        return df.head(n).to_dict(orient="records")
    except Exception:
        return []


def build_llm_payload(run_dir: Path, model_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload["run_dir"] = str(run_dir)
    payload["run_id"] = run_dir.name

    # best summary
    best_path = run_dir / "best_model_summary.txt"
    payload["best_model_summary"] = _load_json_safe(best_path) if best_path.exists() else None

    # per-model metrics from models/*.txt
    models_dir = run_dir / "models"
    payload["model_reports"] = _read_model_txts(models_dir)

    # optional global artifacts
    artifacts = {}
    artifact_paths = (
        "artifacts/metrics/regression_metrics.json",
        "artifacts/metrics/classification_metrics.json",
        "artifacts/explain/shap_top_reasons.csv",
        "artifacts/predictions/regression_predictions.csv",
        "artifacts/predictions/classification_predictions.csv",
        "artifacts/scenario/scenario_results.csv",
    )
    for name in artifact_paths:
        p = model_root / name
        artifacts[name] = None
        if p.exists():
            if p.suffix.lower() == ".json":
                artifacts[name] = _load_json_safe(p)
            elif p.suffix.lower() == ".csv":
                sample_n = 1500 if name == "artifacts/scenario/scenario_results.csv" else 20
                artifacts[name] = _read_csv_sample(p, n=sample_n)
            else:
                artifacts[name] = p.read_text(encoding="utf-8")
    payload["artifacts"] = artifacts

    # include small prediction sample from artifacts or run-specific predictions
    sample_preds = _read_csv_sample(model_root / "artifacts/predictions/regression_predictions.csv", n=10)
    if not sample_preds:
        sample_preds = _read_csv_sample(run_dir / "predictions" / "regression_predictions.csv", n=10)
    payload["prediction_sample"] = sample_preds

    # craft a default prompt template that can be sent to an LLM
    prompt = (
        "You are an assistant that summarizes model runs for environmental monitoring. "
        "Given the run summary, per-model metrics, a few prediction examples, and scenario records, "
        "produce: (1) a one-paragraph executive summary, (2) a table of best models with why they were chosen, "
        "(3) for each site in the sample, recommended actions and short rationale. "
        "Return a JSON object with keys: executive_summary, model_table, site_recommendations.\n\n"
        "Run summary follows. Use the 'best_model_summary' and 'model_reports' fields to compare models. "
    )
    payload["llm_prompt_template"] = prompt

    return payload


def _flatten_model_reports(model_reports: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for model_name, recs in model_reports.items():
        for rec in recs:
            r = {"model_name": model_name}
            r.update(rec)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_payload(run_dir: Path, payload: dict[str, Any]) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[Path] = []

    out_json = run_dir / "llm_payload.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_files.append(out_json)

    out_md = run_dir / "llm_prompt.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# LLM Prompt and Run Summary\n\n")
        f.write("## Prompt Template\n\n")
        f.write(payload.get("llm_prompt_template", ""))
        f.write("\n\n---\n\n")
        f.write("## Best Model Summary\n\n")
        f.write(json.dumps(payload.get("best_model_summary"), ensure_ascii=False, indent=2))
    out_files.append(out_md)

    # write human-readable model summary CSV and markdown table
    models_df = _flatten_model_reports(payload.get("model_reports", {}))
    if not models_df.empty:
        csv_path = run_dir / "llm_models_summary.csv"
        md_path = run_dir / "llm_models_summary.md"
        models_df.to_csv(csv_path, index=False)
        try:
            md_text = models_df.to_markdown(index=False)
        except Exception:
            md_text = models_df.to_string(index=False)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_text)
        out_files.extend([csv_path, md_path])

    # save prediction sample CSV if present
    preds = payload.get("prediction_sample", [])
    if preds:
        pred_df = pd.DataFrame(preds)
        pred_csv = run_dir / "llm_prediction_sample.csv"
        pred_df.to_csv(pred_csv, index=False)
        out_files.append(pred_csv)

    # save scenario records if available in artifacts
    scen_key = "artifacts/scenario/scenario_results.csv"
    scen = payload.get("artifacts", {}).get(scen_key)
    if scen:
        try:
            scen_df = pd.DataFrame(scen).head(1500)
            scen_path = run_dir / "llm_scenario_sample.csv"
            scen_df.to_csv(scen_path, index=False)
            out_files.append(scen_path)
        except Exception:
            pass

    return out_files


def call_openai_chat(prompt_text: str, api_key_env: str = "OPENAI_API_KEY", model: str = "gpt-4o-mini", temperature: float = 0.0, run_dir: Path | None = None) -> dict[str, Any]:
    """
    Call OpenAI Chat Completions API (best-effort). API key is read from env var `api_key_env`.
    This function is optional; callers must ensure the environment provides the API key.
    The raw JSON response is returned and optionally saved to `run_dir/llm_response.json`.
    """
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"OpenAI API key not found in env var {api_key_env}")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": "You are a helpful assistant that summarizes ML model runs and recommends actions."}, {"role": "user", "content": prompt_text}]
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "max_tokens": 1024}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if run_dir is not None:
        try:
            (run_dir / "llm_response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return result


def main(run_id: str | None = None, call_openai: bool = False, openai_model: str = "gpt-4o-mini", openai_temp: float = 0.0, openai_key_env: str = "OPENAI_API_KEY"):
    base = Path(__file__).parents[1]
    output_root = base / "output"
    run_dir = None
    if run_id:
        candidate = output_root / run_id
        if candidate.exists():
            run_dir = candidate
    if run_dir is None:
        run_dir = _find_latest_run(output_root)
    if run_dir is None:
        print("No run directories found under:", output_root)
        return

    payload = build_llm_payload(run_dir, base)
    out_files = save_payload(run_dir, payload)
    print("Saved LLM payload and summaries:")
    for p in out_files:
        print(" -", p)

    if call_openai:
        prompt_text = payload.get("llm_prompt_template", "") + "\n\n" + json.dumps({"best_model_summary": payload.get("best_model_summary"), "model_reports": payload.get("model_reports")}, ensure_ascii=False)
        try:
            result = call_openai_chat(prompt_text, api_key_env=openai_key_env, model=openai_model, temperature=openai_temp, run_dir=run_dir)
            print("OpenAI call succeeded; response saved to llm_response.json")
        except Exception as e:
            print("OpenAI call failed:", e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LLM payload from latest run outputs and optionally call OpenAI")
    parser.add_argument("--run", help="specific run folder name (e.g. run_20260505_141919)", default=None)
    parser.add_argument("--call-openai", action="store_true", help="Call OpenAI Chat API with the generated prompt (requires OPENAI_API_KEY env var)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model to call")
    parser.add_argument("--openai-temp", type=float, default=0.0, help="Temperature for OpenAI call")
    parser.add_argument("--openai-key-env", default="OPENAI_API_KEY", help="Environment variable name that stores OpenAI API key")
    args = parser.parse_args()
    main(args.run, call_openai=args.call_openai, openai_model=args.openai_model, openai_temp=args.openai_temp, openai_key_env=args.openai_key_env)
