action_to_idx = {'food_flip': 0,
                 'food_drizzle': 1,
                 'food_place': 2,
                 'food_remove': 3,
                 'fries_cooking': 4,
                 'fries_serving': 5,
                 'food_packaging': 6,
                 'package_serving': 7,
                 'food_sauce': 8,
                 'food_warm_start': 9,
                 'food_warm_end': 10}
idx_to_action = {v: k for k, v in action_to_idx.items()}

object_to_idx = {'chicken_half': 0, 'chicken_quarter': 1, 'dumpling': 2}
idx_to_object = {v: k for k, v in object_to_idx.items()}
