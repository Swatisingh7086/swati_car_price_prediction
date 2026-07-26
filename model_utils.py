def cast_to_float(x):
    """Casts a column (arrives as strings from HTML form data) to float."""
    return x.astype(float)
