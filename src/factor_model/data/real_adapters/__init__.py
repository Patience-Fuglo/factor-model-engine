"""Real vendor data adapters -- later stage.

When implemented, a real adapter's only job is to populate
``MarketObservation``/``FundamentalObservation`` correctly: real
``known_as_of`` from the vendor's actual publication/filing timestamp
wherever available, with ``quality_flags`` noting any case where a real
timestamp isn't available and a fallback (e.g. a configured publication
lag) had to be used instead. Nothing about the schema changes to
support a real adapter -- see ``data/schemas.py``.
"""
