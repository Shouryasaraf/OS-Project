"""Frozen model comparisons on sampled real held-out candidate decisions."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .model import features


def evaluate_models(views, index, ranker, model_dir: str | Path, output_dir: str | Path,
                    max_decisions: int = 10000) -> list[dict]:
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    models = {
        'logistic_full': ranker,
        'logistic_no_context': joblib.load(model_dir / 'ranker_no_context.joblib'),
        'hist_gradient_boosting_diagnostic': joblib.load(model_dir / 'hist_gradient_boosting.joblib'),
    }
    results = []
    for split, label in ((2, 'validation'), (3, 'test')):
        sample = views.loc[views.split.eq(split)]
        if len(sample) > max_decisions:
            sample = sample.iloc[np.linspace(0, len(sample)-1, max_decisions, dtype=np.int64)]
        matrices, truth, groups, transition_hits = [], [], [], 0
        eligible = 0
        coverage = 0
        for row in sample.itertuples(index=False):
            candidates = index.candidates(int(row.itemid))
            groups.append((len(matrices), len(candidates), bool(row.eligible)))
            if bool(row.eligible):
                eligible += 1
                coverage += int(row.next_item in [item for item,_ in candidates])
                transition_hits += int(bool(candidates) and candidates[0][0] == row.next_item)
            for rank,(item,count) in enumerate(candidates):
                matrices.append(features(row,item,count,rank,index))
                truth.append(int(bool(row.eligible) and item == row.next_item))
        x=np.asarray(matrices,dtype=np.float32)
        y=np.asarray(truth,dtype=np.int8)
        for name,model in models.items():
            matrix=x.copy() if name == 'logistic_no_context' else x
            if name == 'logistic_no_context': matrix[:,4:8] = 0
            prob=model.predict_proba(matrix)[:,1]
            top1=0
            for start,size,is_eligible in groups:
                if size and is_eligible:
                    top1 += int(y[start+int(np.argmax(prob[start:start+size]))])
            results.append({'split':label,'model':name,'sample_decisions':len(sample),
                'eligible_pairs':eligible,'candidate_pool_coverage':coverage/eligible,
                'top1_per_eligible':top1/eligible,'candidate_log_loss':float(log_loss(y,prob))})
        results.append({'split':label,'model':'transition_top1_no_ranker',
            'sample_decisions':len(sample),'eligible_pairs':eligible,
            'candidate_pool_coverage':coverage/eligible,
            'top1_per_eligible':transition_hits/eligible,'candidate_log_loss':None})
    pd.DataFrame(results).to_csv(output_dir / 'model_diagnostics.csv',index=False)
    return results
