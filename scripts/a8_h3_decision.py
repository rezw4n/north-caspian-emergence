"""A8: the PREREG section 10 H3 decision (caspian.h3) from the frozen emergence table,
applied once after the H3 run. Primary: the fused surface with the primary error-model
form against the uncorrected product, the self-referenced emergence functional (D-081).
Verdict per product and region over the near-term scenarios (design.NEAR_TERM_RANGE_M).

Outputs (committed): results/emergence/<tag>/h3_decision.csv, h3_summary.csv, h3_decision.json
Usage: python scripts/a8_h3_decision.py <tag>
"""

import json
import sys

import caspian  # noqa: F401
import pandas as pd

from caspian import exposure as ex
from caspian import h3
from caspian.config import RESULTS


def main(tag: str) -> int:
    d = RESULTS / "emergence" / tag
    # the run's final (fused) form: the primary, or a D-081 sensitivity form (D-096)
    final = (json.loads((d / "run.json").read_text(encoding="utf-8")).get("forms") or [ex.PRIMARY_FORM])[0]
    e = pd.read_csv(d / "emergence.csv")
    e = e[((e["surface"] == "uncorrected") & (e["form"] == "none")) |
          ((e["surface"] == "fused") & (e["form"] == final))]
    dec = h3.decision(e)
    summ = h3.summary(dec)
    dec.to_csv(d / "h3_decision.csv", index=False)
    summ.to_csv(d / "h3_summary.csv", index=False)
    dom = summ[(summ["region"] == "domain") & (summ["functional"] == "emergence")]
    out = {"tag": tag, "primary": f"fused ({final}) vs uncorrected, functional emergence",
           "domain": {r.product: {"verdict": r.verdict, "relevant_scenarios": r.scenarios_relevant,
                                  "n_relevant": int(r.n_relevant), "n_near_term": int(r.n_near_term)}
                      for r in dom.itertuples()}}
    (d / "h3_decision.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    print(summ[summ["functional"] == "emergence"].pivot(index="region", columns="product", values="verdict").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
