"""
Snapshot DAO Governance Pilot Analysis
======================================
Pulls voting data from Snapshot's GraphQL API for several major DAOs,
computes governance concentration metrics (Gini coefficient, top-10
voter share), and produces a comparison chart.

Author: DAO Governance Research Project
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SNAPSHOT_API = "https://hub.snapshot.org/graphql"

# DAOs we want to analyse – keys are human-readable names, values are
# the Snapshot "space" identifiers.
DAOS = {
    "ENS": "ens.eth",
    "Gitcoin": "gitcoindao.eth",
    "Uniswap": "uniswapgovernance.eth",
    "Aave": "aave.eth",
    "Arbitrum": "arbitrumfoundation.eth",
    "Optimism": "opcollective.eth",
    "Lido": "lido-snapshot.eth",
}

# How many proposals per DAO (closed, with >= 10 votes)
PROPOSALS_PER_DAO = 5

# How many votes to fetch per proposal (upper bound)
VOTES_PER_PROPOSAL = 1000

# Paths (relative to the project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run_query(query: str, variables: dict | None = None) -> dict:
    """Send a GraphQL query to the Snapshot API and return the JSON response."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(SNAPSHOT_API, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def gini_coefficient(values: list[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.

    A Gini of 0 means perfect equality (everyone has the same voting power),
    while 1 means perfect inequality (one voter holds all the power).
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    # Standard formula using the sorted array
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


def top_k_share(values: list[float], k: int = 10) -> float:
    """Return the fraction of total value held by the top-k entries."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr_sorted = np.sort(arr)[::-1]  # descending
    top_sum = arr_sorted[:k].sum()
    return top_sum / arr.sum()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_proposals(space_id: str) -> list[dict]:
    """Fetch the most recent closed proposals with at least 10 votes."""
    query = """
    query ($space: String!) {
      proposals(
        where: { space: $space, state: "closed" }
        orderBy: "end"
        orderDirection: desc
        first: 20
      ) {
        id
        title
        votes
        end
        space { id name }
      }
    }
    """
    data = run_query(query, {"space": space_id})
    proposals = data.get("data", {}).get("proposals", [])
    # Keep only those with >= 10 votes, then take the first N
    filtered = [p for p in proposals if p["votes"] >= 10]
    return filtered[:PROPOSALS_PER_DAO]


def fetch_votes(proposal_id: str) -> list[dict]:
    """Fetch individual votes (with voting power) for a proposal."""
    all_votes = []
    skip = 0
    while skip < VOTES_PER_PROPOSAL:
        query = """
        query ($proposal: String!, $first: Int!, $skip: Int!) {
          votes(
            where: { proposal: $proposal }
            first: $first
            skip: $skip
            orderBy: "vp"
            orderDirection: desc
          ) {
            voter
            vp
            choice
            created
          }
        }
        """
        variables = {"proposal": proposal_id, "first": 100, "skip": skip}
        data = run_query(query, variables)
        votes = data.get("data", {}).get("votes", [])
        if not votes:
            break
        all_votes.extend(votes)
        skip += len(votes)
        # Be polite to the API
        time.sleep(0.3)
    return all_votes


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Snapshot DAO Governance Pilot Analysis")
    print("=" * 60)

    all_rows = []        # One row per vote  (raw data)
    summary_rows = []    # One row per proposal (metrics)

    for dao_name, space_id in DAOS.items():
        print(f"\n--- {dao_name} ({space_id}) ---")
        proposals = fetch_proposals(space_id)
        if not proposals:
            print(f"  No qualifying proposals found for {dao_name}.")
            continue

        print(f"  Found {len(proposals)} qualifying proposal(s).")

        for prop in proposals:
            prop_id = prop["id"]
            prop_title = prop["title"][:80]  # truncate long titles
            print(f"  Fetching votes for: {prop_title}...")

            votes = fetch_votes(prop_id)
            if not votes:
                print("    (no votes returned, skipping)")
                continue

            # Collect voting-power values
            vp_values = [v["vp"] for v in votes if v["vp"] is not None]

            # Compute metrics
            gini = gini_coefficient(vp_values)
            top10 = top_k_share(vp_values, k=10)
            total_vp = sum(vp_values)
            n_voters = len(vp_values)

            summary_rows.append({
                "dao": dao_name,
                "space_id": space_id,
                "proposal_id": prop_id,
                "proposal_title": prop_title,
                "num_voters": n_voters,
                "total_voting_power": round(total_vp, 2),
                "gini_coefficient": round(gini, 4),
                "top10_share": round(top10, 4),
            })

            # Store every individual vote for the raw dataset
            for v in votes:
                all_rows.append({
                    "dao": dao_name,
                    "space_id": space_id,
                    "proposal_id": prop_id,
                    "proposal_title": prop_title,
                    "voter": v["voter"],
                    "voting_power": v["vp"],
                    "choice": v["choice"],
                    "vote_timestamp": v["created"],
                })

            # Brief pause between proposals
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------------
    raw_df = pd.DataFrame(all_rows)
    raw_path = os.path.join(DATA_DIR, "dao_governance_data.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw vote data saved to {raw_path}  ({len(raw_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(DATA_DIR, "dao_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}  ({len(summary_df)} rows)")

    # ------------------------------------------------------------------
    # Print the summary table in the terminal
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Per-proposal summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Build a comparison chart (one bar group per DAO)
    # ------------------------------------------------------------------
    if summary_df.empty:
        print("No data to plot.")
        return

    # Aggregate metrics per DAO (mean across proposals)
    agg = summary_df.groupby("dao").agg(
        avg_gini=("gini_coefficient", "mean"),
        avg_top10=("top10_share", "mean"),
        avg_voters=("num_voters", "mean"),
    ).reindex([name for name in DAOS.keys() if name in summary_df["dao"].values])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bar_color = "#4C72B0"

    # Panel 1 – Gini coefficient
    axes[0].barh(agg.index, agg["avg_gini"], color=bar_color)
    axes[0].set_xlabel("Gini Coefficient (0 = equal, 1 = concentrated)")
    axes[0].set_title("Voting Power Inequality")
    axes[0].set_xlim(0, 1)

    # Panel 2 – Top-10 share
    axes[1].barh(agg.index, agg["avg_top10"] * 100, color="#DD8452")
    axes[1].set_xlabel("Top-10 Voter Share (%)")
    axes[1].set_title("Top-10 Concentration")
    axes[1].set_xlim(0, 100)

    # Panel 3 – Average voters per proposal
    axes[2].barh(agg.index, agg["avg_voters"], color="#55A868")
    axes[2].set_xlabel("Average Voters per Proposal")
    axes[2].set_title("Voter Participation")

    fig.suptitle("DAO Governance Pilot – Snapshot Data", fontsize=14, y=1.02)
    fig.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "dao_governance_pilot.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {chart_path}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
