![BFair Logo](https://bfair-ml.github.io/bfair-banner.jpg)

BFair is a Python library designed to automate fair and unbiased classification for various tasks, without requiring extensive machine learning knowledge. It leverages AutoML capabilities (through [AutoGOAL](https://autogoal.github.io)) to promote accessibility and democratization of fairness techniques in machine learning. All methods within BFair have been evaluated and documented for the scientific community.

## 🧰 Features

### 1. Fairness Definitions

BFair provides access to widely recognized fairness definitions commonly used in decision-making and classification tasks. These include: _statistical parity_, _equality of opportunity_, _equalized odds_, and _disparity-based measures_.

### 2. Bias Quantification Metrics

BFair includes a range of metrics for bias quantification, both for generative models and for analyzing bias in the datasets used for training. Specific tools for assessing gender bias in textual data collections are available, enabling comparisons between synthetic and real-world text datasets to understand implicit societal biases.

### 3. Ensemble and Diversification Methods

The library supports multiple ensemble methods, from simple majority-vote classifiers to stacking-based machine learning models. Users can also apply optimization algorithms to determine the optimal configurations for dataset ensembling. Techniques for generating diverse hypothesis models are included to create rich and varied ensembles.

### 4. Bias Mitigation Techniques

BFair offers an easy-to-use interface to automate classification tasks while mitigating biases and promoting fairness. The interface aligns closely with AutoGOAL for smooth interoperability. The AutoML nature of BFair allows users to mitigate bias without requiring extensive knowledge in machine learning.

### 5. Framework for Automatic Annotation of Protected Attributes

BFair provides a framework for automated protected attribute annotation, with built-in sensors, handlers, and domain-specific adjustment algorithms. This framework integrates naturally with BFair's bias mitigation tools and can also be used as a preprocessing step to extract protected attributes for use in other tools.

## ⭐ Quickstart

To illustrate the simplicity of its use, we will load a dataset and run an automatic process in it.

```python
from bfair.methods.autogoal import AutoGoalMitigator
from bfair.metrics import statistical_parity
from bfair.utils import encode_features

from autogoal.kb import MatrixContinuousDense
from autogoal.search import ConsoleLogger, ProgressLogger


# ==============================================================================
# LOAD DATASET
# ==============================================================================
#

dataset = load_adult()
data = dataset.data
feature_names = data.columns[:-1]
target_name = data.columns[-1]
X_train, y_train, encoders = encode_features(
    data, target=target_name, feature_names=feature_names
)

#
# ==============================================================================


# ==============================================================================
# FIND SOLUTION
# ==============================================================================
#

protected_attributes = ["sex"]
attr_indexes = [data.columns.get_loc(attrib) for attrib in protected_attributes]
positive_target = encoders[target_name].transform([">50K"]).item()

mitigator = AutoGoalMitigator.build(
    input=MatrixContinuousDense,
    n_classifiers=5,
    detriment=20,
    pop_size=10,
    search_iterations=1,
    protected_attributes=protected_attributes,
    fairness_metrics=statistical_parity,
    target_attribute=target_name,
    positive_target=">50K",
    sensor=(lambda X: X[:, attr_indexes]),
)
model = mitigator(X_train, y_train, logger=[ProgressLogger(), ConsoleLogger()])

#
# ==============================================================================
```

Sensible defaults are defined for each of BFair's many parameters, ensuring ease of use out of the box.
Most parameters are also compatible with the AutoGOAL backend, allowing seamless integration between the two libraries.

## 🤖 Additional Resources

BFair includes supplemental contributions derived from the research and resources developed to evaluate the framework:

- **Knowledge Extraction Corpus Extension System**: This system enables automatic extension of text corpora with annotated entities and relationships. By integrating multiple annotated versions of the corpus, the system combines their strengths and weaknesses to produce a single, robust version with minimized biases. This allows for calculating disagreement metrics to assess annotation certainty.
  
- **Extended eHealth-KD 2019 Corpus**: Available online for research purposes, this corpus features 8,000 sentences in the medical domain, annotated with four types of entities and thirteen types of relationships. A total of 86,112 elements are annotated, including 50,249 entities and 35,863 relationships. [DOI: 10.5281/zenodo.3926746](https://doi.org/10.5281/zenodo.3926746).

- **Reviews' Gender Corpus**: A corpus containing 70 annotated IMDb reviews. Each review is labeled to indicate any association with gender (men, women, both, or none), and includes a positive or negative sentiment label. [DOI: 10.5281/zenodo.8113901](https://doi.org/10.5281/zenodo.8113901).

## 📃 Publications

The technologies and theoretical results leading up to BFair have been presented at different venues:

- [Automatic extension of corpora from the intelligent ensembling of eHealth knowledge discovery systems outputs](https://doi.org/10.1016/j.jbi.2021.103716). This publication presents an ensemble-based technology and search algorithms to build more robust solutions. The proposed method is applicable to knowledge extraction domains and can be used for the automatic extension of domain resources, provided there is a base collection with diverse annotation versions.

- [Intelligent ensembling of auto-ML system outputs for solving classification problems](https://doi.org/10.1016/j.ins.2022.07.061). This work introduces an Auto-ML component to enable robust automation of arbitrary classification tasks. The capability to generate multiple machine learning architectures that solve the target problem removes the need for a pre-constructed diverse solution set as input.

- [Bias mitigation for fair automation of classification tasks](https://doi.org/10.1111/exsy.13734) This study adds a multi-objective optimization component along with fairness constraints. This enhancement enables fair automation of classification tasks.

- [Automatic annotation of protected attributes to support fairness optimization](https://doi.org/10.1016/j.ins.2024.120188). This work introduces an automatic annotation component for protected attributes, extending the technology’s applicability to scenarios where protected attributes are not explicitly annotated.

- [A multifaceted approach to detect gender biases in Natural Language Generation](https://doi.org/10.1016/j.knosys.2024.112367). This publication introduces a component to quantify biases in text corpora and outputs from large language models.