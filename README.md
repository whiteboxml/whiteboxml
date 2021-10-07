<div style="text-align: center">
    <a href="https://whiteboxml.com">
        <img alt="whiteboxml logo" 
        width="380px" 
        height="140px" 
        src="https://whitebox-public.s3-eu-west-1.amazonaws.com/black_bg_white.svg">
    </a>
</div>

------------------------------------------------------

# WhiteBox Utilities Toolkit: Tools to make your life easier

Fancy data functions that will make your life as a data scientist easier.

## Installing

To install this library in your Python environment:

* `pip install whiteboxml`

## Documentation

### Metrics

#### Classification

* ROC curve / AUC:

```python
import numpy as np

from whiteboxml.modeling.metrics import plot_roc_auc_binary

y_pred = np.random.normal(0, 1, 1000)
y_true = np.random.choice([0, 1], 1000)

ax, fpr, tpr, thr, auc_score = plot_roc_auc_binary(y_pred=y_pred, y_true=y_true, figsize=(8, 8))

ax.get_figure().savefig('roc_curve.png')
```

<img src="https://github.com//whiteboxml/whiteboxml/raw/main/docs/images/roc_auc.png" alt="roc_auc">

* Confusion Matrix:

```python
import numpy as np

from whiteboxml.modeling.metrics import plot_confusion_matrix

y_true = np.random.choice([0, 1, 2, 3], 10000)
y_pred = np.random.choice([0, 1, 2, 3], 10000)

ax, matrix = plot_confusion_matrix(y_pred=y_pred, y_true=y_true, 
                                   class_labels=['a', 'b', 'c', 'd'])

ax.get_figure().savefig('confusion_matrix.png')
```

<img src="https://github.com//whiteboxml/whiteboxml/raw/main/docs/images/confusion_matrix.png" alt="confusion_matrix">

* Optimal Threshold:

```python
import numpy as np

from whiteboxml.modeling.metrics import get_optimal_thr

y_pred_proba = np.random.normal(0, 1, (100, 1))
y_true = np.random.choice([0, 1], (100, 1))

thr = get_optimal_thr(y_pred=y_pred_proba, y_true=y_true)
```
