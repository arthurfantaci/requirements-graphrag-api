#!/usr/bin/env python
"""Compare LangSmith and MLflow observability platforms.

Runs comparison experiments and generates a detailed report on the
differences between the two platforms across multiple dimensions.

Usage:
    # Generate feature comparison
    uv run python scripts/compare_platforms.py --features

    # Run side-by-side tracking experiment
    uv run python scripts/compare_platforms.py --experiment

    # Get platform recommendation
    uv run python scripts/compare_platforms.py --recommend

    # Full comparison (all of the above)
    uv run python scripts/compare_platforms.py --full
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src and repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_feature_comparison() -> None:
    """Print feature comparison between platforms."""
    from jama_mcp_server_graphrag.observability_comparison import (
        compare_platform_features,
    )

    print_header("FEATURE COMPARISON: LangSmith vs MLflow")

    comparison = compare_platform_features()

    # Print feature table
    print("\n{:<35} {:^12} {:^12}".format("Feature", "LangSmith", "MLflow"))
    print("-" * 70)

    for feature in comparison["features"]:
        ls = "✓" if feature["langsmith"] else "✗"
        ml = "✓" if feature["mlflow"] else "✗"
        print(f"{feature['name']:<35} {ls:^12} {ml:^12}")

    # Print summary
    summary = comparison["summary"]

    print("\n" + "-" * 70)
    print("\nLangSmith Advantages:")
    for adv in summary["langsmith_advantages"]:
        print(f"  • {adv}")

    print("\nMLflow Advantages:")
    for adv in summary["mlflow_advantages"]:
        print(f"  • {adv}")

    print("\nBoth Support:")
    for feat in summary["both_support"]:
        print(f"  • {feat}")


def print_recommendation(  # noqa: PLR0913
    needs_self_hosting: bool = False,
    needs_trace_viz: bool = True,
    needs_prompt_hub: bool = False,
    needs_cost_tracking: bool = True,
    budget_sensitive: bool = False,
    langchain_native: bool = True,
) -> None:
    """Print platform recommendation based on requirements."""
    from jama_mcp_server_graphrag.observability_comparison import recommend_platform

    print_header("PLATFORM RECOMMENDATION")

    print("\nYour Requirements:")
    print(f"  • Self-hosting required: {needs_self_hosting}")
    print(f"  • Trace visualization needed: {needs_trace_viz}")
    print(f"  • Prompt hub integration: {needs_prompt_hub}")
    print(f"  • Cost tracking needed: {needs_cost_tracking}")
    print(f"  • Budget sensitive: {budget_sensitive}")
    print(f"  • Using LangChain/LangGraph: {langchain_native}")

    recommendation = recommend_platform(
        needs_self_hosting=needs_self_hosting,
        needs_trace_visualization=needs_trace_viz,
        needs_prompt_hub=needs_prompt_hub,
        needs_cost_tracking=needs_cost_tracking,
        budget_sensitive=budget_sensitive,
        langchain_native=langchain_native,
    )

    print(f"\n>>> Recommended Platform: {recommendation.recommended.value.upper()}")
    print(f"    Confidence: {recommendation.confidence:.0%}")

    print("\nReasons:")
    for reason in recommendation.reasons:
        print(f"  • {reason}")

    if recommendation.considerations:
        print("\nConsiderations:")
        for consideration in recommendation.considerations:
            print(f"  • {consideration}")


async def run_tracking_experiment() -> dict:
    """Run a side-by-side tracking experiment."""
    from jama_mcp_server_graphrag.observability_comparison import (
        Platform,
        UnifiedTracker,
    )

    print_header("TRACKING EXPERIMENT")

    print("\nRunning side-by-side tracking experiment...")

    results = {
        "langsmith": {"success": False, "run_id": None, "error": None},
        "mlflow": {"success": False, "run_id": None, "error": None},
    }

    try:
        # Test tracking to both platforms
        with UnifiedTracker(
            platforms=[Platform.LANGSMITH, Platform.MLFLOW],
            run_name="platform-comparison-test",
            experiment_name="platform-comparison",
        ) as tracker:
            # Log test parameters
            tracker.log_params(
                {
                    "experiment_type": "platform_comparison",
                    "model": "gpt-4o",
                    "similarity_k": 6,
                }
            )

            # Log test metrics
            tracker.log_metrics(
                {
                    "test_faithfulness": 0.85,
                    "test_relevancy": 0.90,
                    "test_precision": 0.80,
                    "test_recall": 0.75,
                }
            )

            # Get results
            for result in tracker.get_results():
                platform_name = result.platform.value
                results[platform_name] = {
                    "success": result.success,
                    "run_id": result.run_id,
                    "error": result.error,
                }

    except Exception as e:
        logger.exception("Experiment failed")
        results["error"] = str(e)

    # Print results
    print("\nExperiment Results:")
    print("-" * 50)

    for platform, data in results.items():
        if platform == "error":
            continue
        status = "✓ Success" if data["success"] else "✗ Failed"
        print(f"\n{platform.upper()}:")
        print(f"  Status: {status}")
        if data["run_id"]:
            print(f"  Run ID: {data['run_id']}")
        if data["error"]:
            print(f"  Error: {data['error']}")

    return results


def print_setup_comparison() -> None:
    """Print setup complexity comparison."""
    print_header("SETUP COMPLEXITY COMPARISON")

    print("\n--- LangSmith Setup ---")
    print("""
1. Sign up at smith.langchain.com
2. Create API key
3. Set environment variables:
   export LANGSMITH_API_KEY=<your-key>
   export LANGSMITH_PROJECT=<project-name>
   export LANGSMITH_TRACING=true
4. Done! LangChain auto-traces.

Time: ~5 minutes
Cost: Free tier available, usage-based pricing
""")

    print("\n--- MLflow Setup ---")
    print("""
1. Install MLflow:
   pip install mlflow
2. Start tracking server:
   mlflow server --host 0.0.0.0 --port 5000
3. Set tracking URI:
   export MLFLOW_TRACKING_URI=http://localhost:5000
4. Create experiment:
   mlflow experiments create -n my-experiment
5. Instrument code manually with mlflow.log_*

Time: ~30 minutes (basic), hours (production)
Cost: Free (open source), infrastructure costs for hosting
""")


def print_cost_comparison() -> None:
    """Print cost comparison."""
    print_header("COST COMPARISON")

    print("""
--- LangSmith Pricing ---
• Free Tier: 5,000 traces/month
• Developer: $39/month (50K traces)
• Team: $400/month (500K traces)
• Enterprise: Custom pricing

--- MLflow Costs ---
• Software: Free (Apache 2.0)
• Self-hosted infrastructure:
  - Small (dev): ~$50-100/month
  - Medium (team): ~$200-500/month
  - Large (enterprise): $1000+/month
• Databricks managed: Based on DBU usage

--- Cost Recommendation ---
• < 5K traces/month: LangSmith free tier
• 5K-50K traces/month: LangSmith Developer
• Budget-conscious + technical team: Self-hosted MLflow
• Enterprise compliance needs: MLflow self-hosted
• Need minimal ops burden: LangSmith managed
""")


def generate_report(output_path: Path) -> None:
    """Generate a comprehensive comparison report."""
    from jama_mcp_server_graphrag.observability_comparison import (
        compare_platform_features,
        recommend_platform,
    )

    print_header("GENERATING COMPARISON REPORT")

    report = {
        "title": "LangSmith vs MLflow Comparison Report",
        "features": compare_platform_features(),
        "recommendations": {
            "default": recommend_platform().__dict__,
            "self_hosted": recommend_platform(needs_self_hosting=True).__dict__,
            "budget_conscious": recommend_platform(budget_sensitive=True).__dict__,
        },
        "setup": {
            "langsmith": {
                "time_minutes": 5,
                "complexity": "low",
                "steps": 4,
            },
            "mlflow": {
                "time_minutes": 30,
                "complexity": "medium",
                "steps": 5,
            },
        },
    }

    # Convert Platform enum to string
    for rec in report["recommendations"].values():
        rec["recommended"] = rec["recommended"].value

    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare LangSmith and MLflow platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--features", action="store_true", help="Show feature comparison")
    parser.add_argument("--experiment", action="store_true", help="Run tracking experiment")
    parser.add_argument("--recommend", action="store_true", help="Get platform recommendation")
    parser.add_argument("--setup", action="store_true", help="Show setup comparison")
    parser.add_argument("--cost", action="store_true", help="Show cost comparison")
    parser.add_argument("--full", action="store_true", help="Run full comparison")
    parser.add_argument("--report", type=str, help="Generate JSON report to file")

    # Recommendation options
    parser.add_argument("--self-hosting", action="store_true", help="Require self-hosting")
    parser.add_argument("--budget", action="store_true", help="Budget sensitive")

    args = parser.parse_args()

    # Default to features if no options specified
    no_options = not any(
        [
            args.features,
            args.experiment,
            args.recommend,
            args.setup,
            args.cost,
            args.full,
            args.report,
        ]
    )
    if no_options:
        args.features = True

    try:
        if args.full:
            args.features = True
            args.setup = True
            args.cost = True
            args.recommend = True

        if args.features:
            print_feature_comparison()

        if args.setup:
            print_setup_comparison()

        if args.cost:
            print_cost_comparison()

        if args.recommend:
            print_recommendation(
                needs_self_hosting=args.self_hosting,
                budget_sensitive=args.budget,
            )

        if args.experiment:
            asyncio.run(run_tracking_experiment())

        if args.report:
            generate_report(Path(args.report))

    except Exception:
        logger.exception("Comparison failed")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
