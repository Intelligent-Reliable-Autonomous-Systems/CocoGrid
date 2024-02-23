import logging

COLOR_MAP = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'yellow': [1, 1, 0, 1],
    'purple': [0.5, 0, 0.5, 1],
    'grey': [0.6, 0.6, 0.6, 1],
    'orange': [1, 0.5, 0, 1],
    'goal_green': [0, 0.9, 0, 1]
}

DEFAULT_GREY = [0.2, 0.2, 0.2, 1]

def get_color_rgba(color):
    if color in COLOR_MAP:
        return COLOR_MAP[color]
    logging.warn('INVALID COLOR', color)
    return DEFAULT_GREY

def get_light_variation(rgba):
    new_rgba = rgba.copy()
    for i in range(3):
        new_rgba[i] = min(1, rgba[i] * 1.2)
    return new_rgba
