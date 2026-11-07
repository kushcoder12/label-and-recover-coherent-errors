"""Run the full real-data analysis pipeline in order.

    python data_analysis/code/run_all.py

Steps (each writes its own outputs/figures):
  1. audit.py            -> cache/extracted.pkl, outputs/data_audit.json
  2. cb_validation.py    -> outputs/pauli_fidelities.json, figures/fig_cb_validation.*
  3. label_correlation.py-> outputs/coherent_recovery.json, figures/fig_label_correlation.*
"""

import audit
import cb_validation
import label_correlation

if __name__ == "__main__":
    print("=" * 70, "\n[1/3] DATA AUDIT\n", "=" * 70, sep="")
    audit.audit()
    print("\n" + "=" * 70, "\n[2/3] CB VALIDATION (credibility anchor)\n", "=" * 70, sep="")
    cb_validation.main()
    print("\n" + "=" * 70, "\n[3/3] LABEL-CORRELATION COHERENT ESTIMATOR\n", "=" * 70, sep="")
    label_correlation.main()
    print("\nDONE. See the data_analysis section of the top-level README for the summary.")
