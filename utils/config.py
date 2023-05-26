COLORS = [(255, 127, 14, 255), (44, 160, 44, 255), (214, 39, 40, 255),
          (148, 103, 189, 255), (140, 86, 75, 255), (227, 119, 194, 255), (188, 189, 34, 255),
          (23, 190, 207, 255), (31, 119, 180, 255)  # tab10 without gray
          ]

THEMES_COLORS = {'dark_amber.xml': (255, 215, 64),
                 'dark_blue.xml': (68, 138, 255),
                 'dark_cyan.xml': (77, 208, 225),
                 'dark_lightgreen.xml': (139, 195, 74),
                 'dark_pink.xml': (255, 64, 129),
                 'dark_purple.xml': (171, 71, 188),
                 'dark_red.xml': (255, 23, 68),
                 'dark_teal.xml': (29, 233, 182),
                 'dark_yellow.xml': (255, 255, 0),
                 'light_amber.xml': (255, 196, 0),
                 'light_blue.xml': (41, 121, 255),
                 'light_blue_500.xml': (3, 169, 244),
                 'light_cyan.xml': (0, 229, 255),
                 'light_cyan_500.xml': (0, 188, 212),
                 'light_lightgreen.xml': (100, 221, 23),
                 'light_lightgreen_500.xml': (139, 195, 74),
                 'light_orange.xml': (255, 61, 0),
                 'light_pink.xml': (255, 64, 129),
                 'light_pink_500.xml': (233, 30, 99),
                 'light_purple.xml': (224, 64, 251),
                 'light_purple_500.xml': (156, 39, 176),
                 'light_red.xml': (255, 23, 68),
                 'light_red_500.xml': (244, 67, 54),
                 'light_teal.xml': (29, 233, 182),
                 'light_teal_500.xml': (0, 150, 136),
                 'light_yellow.xml': (255, 234, 0)}

FAT_POINT_SCALE = 1.0
FAT_AREA_AROUND = 1.0
ACTIVE_COLOR = (235, 255, 15, 120)  # white
FAT_POINT_COLOR = (0, 0, 0, 255)  # black
POSITIVE_POINT_COLOR = (15, 255, 183, 255)  # green
NEGATIVE_POINT_COLOR = (255, 209, 109, 255)  # red
POLYGON_AREA_THRESHOLD = 100
MIN_POLYGON_MOVE_DISTANCE = 5
DENSITY_SCALE = 50
MIN_DENSITY_VALUE = 0.5
MAX_DENSITY_VALUE = 7

PATH_TO_GROUNDING_DINO_CONFIG = 'gd\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py'
PATH_TO_GROUNDING_DINO_CHECKPOINT = 'gd\groundingdino_swint_ogc.pth'
PATH_TO_SAM_CHECKPOINT = 'sam_models\sam_vit_h_4b8939.pth'

LANGUAGE = 'ENG'  # RU
