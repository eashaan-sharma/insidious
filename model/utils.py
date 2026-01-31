def save_image(file, path):
    """
    Saves uploaded image to disk
    """
    with open(path, "wb") as f:
        f.write(file.read())
