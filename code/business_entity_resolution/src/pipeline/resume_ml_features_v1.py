from pathlib import Path
import sqlite3
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src")
)


# ============================================================
# IMPORT EXISTING PIPELINE
# ============================================================

from pipeline.build_ml_train_v1 import (
    TRAIN_DB,
    OUT_DB,
    make_features,
    write_metadata,
    configure_output,
    attach_output,
    log,
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRAIN_DB.exists():

        raise FileNotFoundError(
            f"Blocking DB not found: "
            f"{TRAIN_DB}"
        )


    if not OUT_DB.exists():

        raise FileNotFoundError(
            f"ML DB not found: "
            f"{OUT_DB}"
        )


    # --------------------------------------------------------
    # Open blocking DB READ-ONLY
    # --------------------------------------------------------

    src = sqlite3.connect(
        TRAIN_DB.resolve().as_uri()
        + "?mode=ro",

        uri=True,

        timeout=300,
    )


    try:

        # ----------------------------------------------------
        # Configure source connection
        # ----------------------------------------------------

        configure_output(
            src
        )


        # ----------------------------------------------------
        # Attach existing ML DB
        # ----------------------------------------------------

        attach_output(
            src,
            OUT_DB
        )


        # ----------------------------------------------------
        # Check existing data
        # ----------------------------------------------------

        feature_count = src.execute(
            """
            SELECT COUNT(*)
            FROM ml.features
            """
        ).fetchone()[0]


        pair_count = src.execute(
            """
            SELECT COUNT(*)
            FROM ml.ml_pairs
            """
        ).fetchone()[0]


        candidate_count = src.execute(
            """
            SELECT COUNT(*)
            FROM ml.candidate_union
            """
        ).fetchone()[0]


        positive_count = src.execute(
            """
            SELECT COUNT(*)
            FROM ml.ml_pairs
            WHERE label = 1
            """
        ).fetchone()[0]


        negative_count = src.execute(
            """
            SELECT COUNT(*)
            FROM ml.ml_pairs
            WHERE label = 0
            """
        ).fetchone()[0]


        log(
            "========================================"
        )

        log(
            "RESUME ML FEATURE GENERATION"
        )

        log(
            "========================================"
        )

        log(
            f"Candidate pairs : "
            f"{candidate_count:,}"
        )

        log(
            f"ML pairs        : "
            f"{pair_count:,}"
        )

        log(
            f"Positive pairs  : "
            f"{positive_count:,}"
        )

        log(
            f"Negative pairs  : "
            f"{negative_count:,}"
        )

        log(
            f"Existing features: "
            f"{feature_count:,}"
        )

        log(
            "========================================"
        )


        # ----------------------------------------------------
        # If features already exist, don't duplicate them
        # ----------------------------------------------------

        if feature_count > 0:

            log(
                "Features already exist."
            )

            log(
                "Nothing to do."
            )

            return


        # ----------------------------------------------------
        # Generate features from existing ML pairs
        # ----------------------------------------------------

        make_features(
            src
        )


    finally:

        try:

            src.execute(
                "DETACH DATABASE ml"
            )

        except Exception:

            pass


        src.close()


    # --------------------------------------------------------
    # Write metadata
    # --------------------------------------------------------

    write_metadata()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()