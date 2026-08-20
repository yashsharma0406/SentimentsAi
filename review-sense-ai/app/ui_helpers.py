def explain(simple, technical, mode):
    return simple if mode == "Simple Language" else technical
def sentiment_band(p):
    if p >= 0.85:
        return "Strong Positive 😊", "Very High"
    elif p >= 0.6:
        return "Positive 🙂", "High"
    elif p >= 0.4:
        return "Mixed 😐", "Medium"
    elif p >= 0.15:
        return "Negative 🙁", "Low"
    else:
        return "Strong Negative 😡", "Very Low"
