"""Generate paper-ready scientific report and SVG charts."""

import _bootstrap  # noqa: F401

from query_agent_eval.scientific_report import build_scientific_report


def main() -> None:
    path = build_scientific_report()
    print(f"Scientific report: {path}")
    print(f"Charts: {path.parent / 'charts'}")


if __name__ == "__main__":
    main()
