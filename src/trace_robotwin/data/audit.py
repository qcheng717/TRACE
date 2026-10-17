import numpy as np
from trace_robotwin.data.schema import ARMS

def calibrate(values):
    """Require a resolved inactive cluster, never force a two-cluster interpretation.

    Exact static trajectories have zero differences. Otherwise require >=100x
    largest gap and >=10% samples on either side to distinguish numerical drift
    from task motion. Gap endpoints form an explicit uncertainty interval.
    """
    x=np.sort(np.asarray(values,dtype=np.float64)); positive=x[x>0]
    result=dict(quantiles=dict(zip(['min','p10','p25','p50','p75','p90','max'],np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist())),zero_count=int((x==0).sum()))
    if len(positive) and (x==0).mean()>=.1 and positive[0]>100*np.finfo(np.float32).eps:
        low,high=0.,float(positive[0]); method='exact-zero inactive cluster; minimum observed positive activity'
    elif len(positive)>2:
        gaps=positive[1:]/positive[:-1]; order=np.argsort(gaps)[::-1]; found=None
        for i in order:
            low,high=float(positive[i]),float(positive[i+1])
            if gaps[i]>=100 and (x<=low).mean()>=.1 and (x>=high).mean()>=.1:
                found=(low,high);break
        if found is None: return dict(**result,resolved=False,reason='No separated inactive/active clusters')
        low,high=found; method='largest >=100x empirical gap, each cluster >=10%'
    else: return dict(**result,resolved=False,reason='No measurable active/static separation')
    return dict(**result,resolved=True,inactive_max=low,active_min=high,threshold=(low+high)/2,method=method)


def classify(row,calibration):
    flags=[]
    for arm in ARMS:
        states=[]
        for suffix,metric in [('motion','joint'),('gripper_activity','gripper')]:
            c=calibration[metric]; v=row[f'{arm}_{suffix}']
            states.append(None if not c['resolved'] else (True if v>=c['active_min'] else False if v<=c['inactive_max'] else None))
        flags.append(True if True in states else False if all(s is False for s in states) else None)
    if flags==[True,False]: return 'left','only left exceeds calibrated activity bounds'
    if flags==[False,True]: return 'right','only right exceeds calibrated activity bounds'
    if flags==[True,True]: return 'both','both arms or grippers move; excluded'
    return 'ambiguous','static or unresolved motion distribution; excluded'


