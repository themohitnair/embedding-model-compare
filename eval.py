import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np

client = AsyncQdrantClient(host="localhost", port=6333)

MODELS = {
    "mpnet": {"name": "all-mpnet-base-v2", "dim": 768, "color": "lightblue"},
    "minilm": {"name": "all-MiniLM-L6-v2", "dim": 384, "color": "lightcoral"},
    "e5_large": {"name": "intfloat/e5-large-v2", "dim": 1024, "color": "lightgreen"},
    "e5_base": {"name": "intfloat/e5-base-v2", "dim": 768, "color": "lightyellow"},
    "bge_large": {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "color": "lightpink"},
    "bge_base": {"name": "BAAI/bge-base-en-v1.5", "dim": 768, "color": "lightgray"},
    "gtr_t5": {
        "name": "sentence-transformers/gtr-t5-large",
        "dim": 768,
        "color": "lightsalmon",
    },
}

SPECIFIC_TOPICS = [
    "England Cricket Team",
    "England Football Team",
    "Black Coffee",
    "Indian Cricket Team",
    "Black History Month",
]


async def evaluate_model(model_key, model, k=100, score_threshold=0.3):
    results = {}
    collection_name = f"{model_key}_posts"

    for topic in SPECIFIC_TOPICS:
        print(f"Querying '{topic}' in {collection_name}...")

        query_vector = model.encode(topic).tolist()

        query_results = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=score_threshold,
        )

        retrieved_topics = [point.payload["topic"] for point in query_results.points]

        true_positives = sum(1 for t in retrieved_topics if t == topic)
        total_retrieved = len(retrieved_topics)
        false_positives = total_retrieved - true_positives

        precision = true_positives / total_retrieved if total_retrieved > 0 else 0

        results[topic] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": true_positives / 100,
            "retrieved_topics": retrieved_topics,
            "total_retrieved": total_retrieved,
        }

        print(
            f"  Retrieved: {total_retrieved}, TP: {true_positives}, FP: {false_positives}"
        )
        print(f"  Precision: {precision:.3f}, Recall: {true_positives / 100:.3f}")

    return results


def plot_all_models_comparison(all_results):
    fig, axes = plt.subplots(
        len(MODELS), len(SPECIFIC_TOPICS), figsize=(25, 4 * len(MODELS))
    )
    fig.suptitle("Topic Distribution Comparison Across All Models", fontsize=16)

    for row, (model_key, model_info) in enumerate(MODELS.items()):
        for col, topic in enumerate(SPECIFIC_TOPICS):
            counter = Counter(all_results[model_key][topic]["retrieved_topics"])
            labels = list(counter.keys())
            counts = list(counter.values())

            axes[row, col].bar(range(len(labels)), counts, color=model_info["color"])
            axes[row, col].set_title(f"{topic}\n({model_key.upper()})", fontsize=9)
            axes[row, col].set_xticks(range(len(labels)))
            axes[row, col].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axes[row, col].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("all_models_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_comprehensive_table(all_results):
    comparison_data = []

    for topic in SPECIFIC_TOPICS:
        row = {"Topic": topic}

        for model_key in MODELS.keys():
            row[f"{model_key.upper()}_Retrieved"] = all_results[model_key][topic][
                "total_retrieved"
            ]
            row[f"{model_key.upper()}_TP"] = all_results[model_key][topic][
                "true_positives"
            ]
            row[f"{model_key.upper()}_FP"] = all_results[model_key][topic][
                "false_positives"
            ]
            row[f"{model_key.upper()}_Precision"] = all_results[model_key][topic][
                "precision"
            ]
            row[f"{model_key.upper()}_Recall"] = all_results[model_key][topic]["recall"]

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def create_summary_table(all_results):
    summary_data = []

    for model_key, model_info in MODELS.items():
        avg_precision = np.mean(
            [all_results[model_key][topic]["precision"] for topic in SPECIFIC_TOPICS]
        )
        avg_recall = np.mean(
            [all_results[model_key][topic]["recall"] for topic in SPECIFIC_TOPICS]
        )
        total_retrieved = sum(
            [
                all_results[model_key][topic]["total_retrieved"]
                for topic in SPECIFIC_TOPICS
            ]
        )
        total_tp = sum(
            [
                all_results[model_key][topic]["true_positives"]
                for topic in SPECIFIC_TOPICS
            ]
        )
        total_fp = sum(
            [
                all_results[model_key][topic]["false_positives"]
                for topic in SPECIFIC_TOPICS
            ]
        )

        summary_data.append(
            {
                "Model": model_key.upper(),
                "Model_Name": model_info["name"],
                "Dimensions": model_info["dim"],
                "Avg_Precision": avg_precision,
                "Avg_Recall": avg_recall,
                "Total_Retrieved": total_retrieved,
                "Total_TP": total_tp,
                "Total_FP": total_fp,
                "Overall_Precision": total_tp / total_retrieved
                if total_retrieved > 0
                else 0,
            }
        )

    return pd.DataFrame(summary_data).sort_values("Avg_Precision", ascending=False)


async def main(k=100, score_threshold=0.3):
    print(
        f"🔍 Evaluating {len(MODELS)} embedding models with k={k}, threshold={score_threshold}"
    )
    print("=" * 80)

    models = {}
    all_results = {}

    for model_key, model_info in MODELS.items():
        print(f"Loading {model_info['name']}...")
        models[model_key] = SentenceTransformer(model_info["name"])

    for model_key, model in models.items():
        print(f"\nEvaluating {MODELS[model_key]['name']}...")
        all_results[model_key] = await evaluate_model(
            model_key, model, k, score_threshold
        )

    print("\n📊 Generating comprehensive comparison charts...")
    plot_all_models_comparison(all_results)

    print("\n📋 Detailed Results:")
    detailed_df = create_comprehensive_table(all_results)
    print(detailed_df.to_string(index=False, float_format="%.3f"))

    print("\n🏆 Model Performance Summary:")
    summary_df = create_summary_table(all_results)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    print(
        f"\n🥇 Winner: {summary_df.iloc[0]['Model']} ({summary_df.iloc[0]['Model_Name']})"
    )
    print(f"   Avg Precision: {summary_df.iloc[0]['Avg_Precision']:.3f}")
    print(f"   Dimensions: {summary_df.iloc[0]['Dimensions']}")

    detailed_df.to_csv("detailed_model_comparison.csv", index=False)
    summary_df.to_csv("model_summary.csv", index=False)
    print("\n💾 Results saved to CSV files and all_models_comparison.png")


if __name__ == "__main__":
    import sys

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    print(f"Using k={k}, score_threshold={score_threshold}")
    asyncio.run(main(k, score_threshold))
