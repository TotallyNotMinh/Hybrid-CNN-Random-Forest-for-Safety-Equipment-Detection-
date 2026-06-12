import os

file_path = 'report.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. Informal phrasing
    ('letting an employer check for PPE compliance manually in a vast space', 
     'relying on manual PPE compliance audits across vast industrial spaces'),
    ('the main idea of our approach is to combine', 
     'the core methodology of our approach combines'),
    
    # 2. Over-dramatic verbs
    ('destroying geometric and texture cues', 'severely attenuating geometric and texture cues'),
    ('destroying geometric shape information', 'severely attenuating geometric shape information'),
    
    # 3. Redundancy
    ('According to the Ministry of Home Affairs of Vietnam, Vietnam recorded', 
     'According to the Vietnamese Ministry of Home Affairs, the country recorded'),
    
    # 4. Standard Terminology (Positive Class -> Target Class)
    ('treated as the \\textbf{positive class}', 'treated as the \\textbf{target class (or positive class)}'),
    ('treating \\texttt{NO-Hardhat} as the positive class', 'treating \\texttt{NO-Hardhat} as the target class'),
    
    # 5. Overuse of real-world (Replacing select instances for variety)
    ('In real-world settings, automating', 'In operational environments, automating'),
    ('\\textbf{O5 (Real-World Validation):}', '\\textbf{O5 (Out-of-Distribution Validation):}'),
    ('unseen real-world images', 'unseen deployment images'),
    ('independent real-world test set', 'independent operational test set'),
    ('restricted size of the real-world deployment dataset', 'restricted size of the deployment dataset'),
    ('real-world images were sourced', 'operational images were sourced'),
    ('real-world deployment requires strict', 'operational deployment requires strict')
]

original_content = content
for old_text, new_text in replacements:
    content = content.replace(old_text, new_text)

if content != original_content:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Replacements applied successfully.')
else:
    print('No changes made. Please check the target strings.')
