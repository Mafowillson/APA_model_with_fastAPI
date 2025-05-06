import re

def extract_features(text):
    return {
        'has_year_parentheses': bool(re.search(r'\(\d{4}\)', text)),
        'has_comma_after_author': bool(re.search(r'[A-Z][a-z]+, [A-Z]\.', text)),
        'has_period_after_initials': bool(re.search(r'[A-Z]\.', text)),
        'has_multiple_periods': text.count('.') >= 3,
        'has_et_al': 'et al.' in text.lower(),
        'has_final_period': text.strip().endswith('.')
    }

def explain_violations(text):
    features = extract_features(text)
    violations = []

    if not features['has_year_parentheses']:
        violations.append("Year is not in parentheses")
    if not features['has_comma_after_author']:
        violations.append("Missing comma after author name")
    if not features['has_period_after_initials']:
        violations.append("Missing period after initials")
    if not features['has_multiple_periods']:
        violations.append("Too few periods (structure might be off)")
    if not features['has_et_al']:
        violations.append("Missing 'et al.' (if multiple authors)")
    if not features['has_final_period']:
        violations.append("Missing full stop at the end")

    return features, violations
