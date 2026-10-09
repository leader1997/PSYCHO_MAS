"""Build and save the 50 stratified HMS case profiles."""

import _bootstrap  # noqa: F401

from query_agent_eval.profiles import build_profiles, save_profiles


def main() -> None:
    records = build_profiles()
    out = save_profiles(records)
    strata: dict[str, int] = {}
    for r in records:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    print(f"Saved {len(records)} profiles -> {out}")
    print(f"Strata: {strata}")


if __name__ == "__main__":
    main()
