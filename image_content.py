from PIL import ImageOps
from PIL import ImageStat


def image_empty(image):
    """
    Reads Pillow image and determines by looking at image statistics if there are enough non white pixels to consider if there is something on the picture.
    
    ---
    image : PIL Image
        PIL Image object.
    """
    gray_image = ImageOps.grayscale(image)
    stats = ImageStat.Stat(gray_image)

    # Trial and error.
    if stats.mean > 245:
        return True

    if stats.var < 250:
        return True

    if stats.stddev < 20:
        return True

    return False
