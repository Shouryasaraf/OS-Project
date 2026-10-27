"""Classifier zoo (track CLF): tuned alternatives to the shipped GNB router.

Part 1 (:mod:`.synthetic`) tunes 13 classifier families plus the two shipped
online models on labelled synthetic windows and scores them once on disjoint
test seeds. Part 2 (:mod:`.router`) plugs every tuned classifier into the
unchanged ``adaptive`` replay mode on real block traces. :mod:`.data` and
:mod:`.router`'s replay helpers are stdlib-only; :mod:`.zoo` imports
scikit-learn lazily. Pre-registration: ``results/classifier/PREREGISTERED.md``.
"""
