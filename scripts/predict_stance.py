#!/usr/bin/env python
"""Command-line script for stance classification using target-conditioned DistilBERT."""

from __future__ import annotations

import sys
from constituency_insights.nlp.predict import main

if __name__ == "__main__":
    sys.exit(main())
