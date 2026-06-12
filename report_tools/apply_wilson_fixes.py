import os

file_path = 'report.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. Update text mentioning Wald interval
    ('(calculated using the Wald interval)', '(calculated using the Wilson score interval)'),
    
    # 2. Update Table 12 confidence intervals
    ('97.4\\% $[94.0\\%, 100.0\\%]$', '97.4\\% $[91.6\\%, 99.2\\%]$'),
    ('96.6\\%* $[92.7\\%, 100.0\\%]$', '96.6\\%* $[90.4\\%, 98.8\\%]$'),
    ('96.6\\% $[92.7\\%, 100.0\\%]$', '96.6\\% $[90.4\\%, 98.8\\%]$'),
    ('97.7\\% $[94.5\\%, 100.0\\%]$', '97.7\\% $[92.0\\%, 99.4\\%]$'),
    ('79.2\\% $[70.7\\%, 87.7\\%]$', '79.2\\% $[69.5\\%, 86.4\\%]$'),
    ('73.6\\% $[64.3\\%, 82.8\\%]$', '73.6\\% $[63.5\\%, 81.7\\%]$'),
    
    # 3. Remove the Wald interval limitation from Threats to Validity
    ('  \\item \\textbf{Wald Interval Limitations (Statistical):} The Wald interval used to calculate deployment confidence intervals assumes normality and can exhibit poor coverage properties for small sample sizes or proportions near 0 or 1, potentially underestimating the uncertainty boundary.\n', '')
]

for old_text, new_text in replacements:
    content = content.replace(old_text, new_text)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Wilson interval updates applied successfully.')
