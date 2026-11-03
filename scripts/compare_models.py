#!/usr/bin/env python
"""Command-line script for paired validation comparison between baseline and DistilBERT."""

from __future__ import annotations

import sys
from constituency_insights.nlp.comparison import main

if __name__ == "__main__":
    sys.exit(main())
