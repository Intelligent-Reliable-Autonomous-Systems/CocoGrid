COLOR_MAP = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'yellow': [1, 1, 0, 1],
    'purple': [0.5, 0, 0.5, 1],
    'grey': [0.6, 0.6, 0.6, 1]
}

DEFAULT_GREY = [0.2, 0.2, 0.2, 1]

def getColorRGBA(color):
    if color in COLOR_MAP:
        return COLOR_MAP[color]
    print('INVALID COLOR', color)
    return DEFAULT_GREY