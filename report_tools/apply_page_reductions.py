import os
import re

file_path = 'report.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove Section 6: Random Forest Classification Theory
section6_pattern = r'\\section\{Random Forest Classification Theory\}.*?(?=\\section\{Classifier Comparison\})'
content = re.sub(section6_pattern, '', content, flags=re.DOTALL)

# 2. Remove Section 10.4: The Representation Bottleneck and Linear Separability
section104_pattern = r'\\subsection\{The Representation Bottleneck and Linear Separability\}.*?(?=\\subsection\{Threats to Validity\})'
content = re.sub(section104_pattern, '', content, flags=re.DOTALL)

# 3. Condense Section 10.3 (and Table 13)
section103_pattern = r'\\subsection\{Shortcut Learning and Feature Importance Concentration\}.*?(?=\\subsection\{Threats to Validity\})'

new_section103 = r"""\subsection{Shortcut Learning and Feature Importance Concentration}
Random Forests produce feature importance scores natively by tracking the mean decrease in Gini impurity across all splits. For our 512-dimensional ResNet18 embeddings, analyzing which dimensions the forest relies on most provides insight into the shortcut learning problem. The highest single feature weight across any task is only 6.88\% (for gloves), indicating the model relies on a distributed representation across many dimensions rather than a single dominant feature. We also observe shared important features across tasks, suggesting that certain dimensions encode visual characteristics common to multiple PPE items.

To quantify this concentration, we calculated the cumulative feature weight for each task. The forest constructs broad, ensemble-wide decision boundaries across a large fraction of the semantic feature space. For the Hardhat and Safety Vest classifiers, nearly half of the feature dimensions (237 and 245 dimensions, respectively) are required to capture 80\% of the decision weight. This highly distributed representation explains why local changes in background context can perturb many features simultaneously and trigger misclassifications.

"""

# Use lambda m: new_section103 to avoid python parsing escape sequences like \s inside the replacement string
content = re.sub(section103_pattern, lambda m: new_section103, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Page reduction edits applied successfully.')
