
LANE_TYPES = {
    0: {
        'name': 'curb',
        'color': (200, 200, 200),
        'double': False,
        'dashed': False,
    },

    1: {
        'name': 'solid_yellow',
        'color': (220, 190, 40),
        'double': False,
        'dashed': False,
    },

    2: {
        'name': 'dashed_yellow',
        'color': (220, 190, 40),
        'double': False,
        'dashed': True,
    },

    3: {
        'name': 'double_solid_yellow',
        'color': (220, 190, 40),
        'double': True,
        'dashed': [False, False],
    },

    4: {
        'name': 'dashed_solid_yellow',
        'color': (220, 190, 40),
        'double': True,
        'dashed': [True, False],
    },

    5: {
        'name': 'solid_dashed_yellow',
        'color': (220, 190, 40),
        'double': True,
        'dashed': [False, True],
    },

    6: {
        'name': 'solid_white',
        'color': (240, 240, 240),
        'double': False,
        'dashed': False,
    },

    7: {
        'name': 'dashed_white',
        'color': (240, 240, 240),
        'double': False,
        'dashed': True,
    },

    8: {
        'name': 'double_solid_white',
        'color': (240, 240, 240),
        'double': True,
        'dashed': [False, False],
    },

    9: {
        'name': 'dashed_solid_white',
        'color': (240, 240, 240),
        'double': True,
        'dashed': [True, False],
    },

    10: {
        'name': 'solid_dashed_white',
        'color': (240, 240, 240),
        'double': True,
        'dashed': [False, True],
    },
}



def get_number(name):
    for number in LANE_TYPES:
        if LANE_TYPES[number]['name'] == name:
            return number
    