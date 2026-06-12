import os

file_path = 'report.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. LaTeX Math Comma Spacing
    ('($N = 158,338$ samples)', '($N = 158{,}338$ samples)'),
    ('($126,670$ train, $31,668$ validation)', '($126{,}670$ train, $31{,}668$ validation)'),
    
    # 2. Removing Inline Bolding in Section 10.2
    ('\\textbf{8} appear', '8 appear'),
    ('\\textbf{3.19$\\times$ dilution factor}', '3.19$\\times$ dilution factor'),
    ('\\textbf{16.56\\%}', '16.56\\%'),
    ('\\textbf{48.55\\%}', '48.55\\%'),
    ('\\textbf{16.19 nodes}', '16.19 nodes'),
    ('\\textbf{17.76 nodes}', '17.76 nodes'),
    ('\\textbf{8.47 levels of splits}', '8.47 levels of splits'),
    ('\\textbf{92.54\\%}', '92.54\\%'),
    ('\\textbf{1.72}', '1.72'),
    ('\\textbf{79.64\\%}', '79.64\\%'),
    ('\\textbf{4.34}', '4.34'),
    ('\\textbf{5.40\\%}', '5.40\\%'),
    ('\\textbf{27.90\\%}', '27.90\\%'),
    ('\\textbf{16.45\\%}', '16.45\\%')
]

original_content = content
for old_text, new_text in replacements:
    content = content.replace(old_text, new_text)

if content != original_content:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Quick fixes applied successfully.')
else:
    print('No changes made. Please check the target strings.')
