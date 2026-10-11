"""Measured rack openings in metres, ordered front row then rear row."""
HOLES_LOCAL = tuple((x, y) for y in (-.0136573, .0136573)
                    for x in (-.0682855, -.0409713, -.0136571, .0136571, .0409713, .0682855))
OCCUPIED_SLOTS = (0, 2, 4, 7, 9)
TUBE_NAMES = ("object", "rack_tube_2", "rack_tube_3", "rack_tube_4", "rack_tube_5")
