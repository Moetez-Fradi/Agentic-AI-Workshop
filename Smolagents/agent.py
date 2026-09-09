import sys

import pandas as pd
from dotenv import load_dotenv

from observability import setup_tracing
from ollama_model import get_model

from smolagents import CodeAgent, tool

DEFAULT_QUESTION = (
    "Using data/marketing_sales.csv (columns: date, region, product, revenue, "
    "marketing_spend, units_sold), find the top 3 underperforming products in "
    "the EMEA region, compute each one's 14-day rolling average revenue, and "
    "plot the three rolling averages on one chart saved as rolling_avg.png. "
    "Explain which products are underperforming and why."
)


@tool
def inspect_sales_dataset(file_path: str = "data/marketing_sales.csv") -> str:
    """Inspect a sales CSV and return a compact overview for analysis.

    Args:
        file_path: Path to the sales CSV file.
    """
    data = pd.read_csv(file_path)
    date_column = pd.to_datetime(data["date"])
    regions = ", ".join(sorted(data["region"].dropna().unique()))
    products = ", ".join(sorted(data["product"].dropna().unique()))

    return (
        f"Rows: {len(data)}; columns: {', '.join(data.columns)}; "
        f"date range: {date_column.min().date()} to {date_column.max().date()}; "
        f"regions: {regions}; products: {products}"
    )


def main() -> None:
    load_dotenv()

    if setup_tracing():
        print("Tracing enabled.\n")
    else:
        print("Tracing disabled.\n")

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    agent = CodeAgent(
        tools=[inspect_sales_dataset],
        model=get_model(),
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    )

    agent.run(question)


if __name__ == "__main__":
    main()
