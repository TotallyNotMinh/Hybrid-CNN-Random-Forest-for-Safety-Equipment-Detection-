import math

def wilson_interval(p, n, z=1.96):
    denominator = 1 + z**2/n
    center_adjusted_prob = p + z**2 / (2*n)
    adjusted_std_dev = math.sqrt((p*(1 - p) / n) + z**2 / (4 * n**2))
    
    lower_bound = (center_adjusted_prob - z * adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + z * adjusted_std_dev) / denominator
    
    return max(0.0, lower_bound * 100), min(100.0, upper_bound * 100)

accuracies = [0.974, 0.966, 0.977, 0.792, 0.736]
n = 87

print('Wilson Intervals:')
for acc in accuracies:
    lower, upper = wilson_interval(acc, n)
    print(f'{acc*100:.1f}% -> [{lower:.1f}%, {upper:.1f}%]')
