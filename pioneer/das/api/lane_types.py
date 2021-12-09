
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

    100: {
        'name': 'solid_blue',
        'color': (0x1f, 0x77, 0xb4),
        'double': False,
        'dashed': False,
    },

    101: {
        'name': 'solid_orange',
        'color': (0xff, 0x7f, 0x0e),
        'double': False,
        'dashed': False,
    },

    102: {
        'name': 'solid_green',
        'color': (0x2c, 0xa0, 0x2c),
        'double': False,
        'dashed': False,
    },

    103: {
        'name': 'solid_red',
        'color': (0xd6, 0x27, 0x28),
        'double': False,
        'dashed': False,
    },

    104: {
        'name': 'solid_purple',
        'color': (0x94, 0x67, 0xbd),
        'double': False,
        'dashed': False,
    },

    105: {
        'name': 'solid_brown',
        'color': (0x8c, 0x56, 0x4b),
        'double': False,
        'dashed': False,
    },

    106: {
        'name': 'solid_pink',
        'color': (0xe3, 0x77, 0xc2),
        'double': False,
        'dashed': False,
    },

    107: {
        'name': 'solid_gray',
        'color': (0x7f, 0x7f, 0x7f),
        'double': False,
        'dashed': False,
    },

    108: {
        'name': 'solid_olive',
        'color': (0xbc, 0xbd, 0x22),
        'double': False,
        'dashed': False,
    },

    109: {
        'name': 'solid_cyan',
        'color': (0x17, 0xbe, 0xcf),
        'double': False,
        'dashed': False,
    },

}



def get_number(name):
    for number in LANE_TYPES:
        if LANE_TYPES[number]['name'] == name:
            return number
    