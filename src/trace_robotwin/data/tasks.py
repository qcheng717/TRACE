"""Fixed 20-task ordering used by the RoboTwin experiment."""
ROBOTWIN_TASKS = [
    'move_can_pot', 'move_pillbottle_pad', 'move_playingcard_away',
    'place_container_plate', 'place_empty_cup', 'place_fan', 'place_mouse_pad',
    'place_object_scale', 'place_shoe', 'beat_block_hammer', 'click_alarmclock',
    'click_bell', 'press_stapler', 'stamp_seal', 'turn_switch', 'dump_bin_bigbin',
    'open_microwave', 'rotate_qrcode', 'shake_bottle', 'shake_bottle_horizontally',
]
TASK_GROUPS = dict(pick_place=ROBOTWIN_TASKS[:9],
                   contact_tool=ROBOTWIN_TASKS[9:15],
                   dynamic_articulated=ROBOTWIN_TASKS[15:])
TASK_IDS = {task: i for i, task in enumerate(ROBOTWIN_TASKS)}
TASK_TO_GROUP = {task: group for group, tasks in TASK_GROUPS.items() for task in tasks}
