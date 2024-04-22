from fobo2_env.src.utils import value_in_range


def calculate_distance_reward(
    distance: int, desired_distance: int, max_distace: int, offset: int
) -> float:
    reward = 0.0

    if value_in_range(
        value=distance,
        center=desired_distance,
        offset=offset,
    ):
        return 5.0

    if distance > desired_distance:
        reward = calculate_linear_score(
            max_score=1,
            max_value=desired_distance,
            min_score=0,
            min_value=max_distace,
            value=distance,
        )
    else:
        reward = calculate_linear_score(
            max_score=1,
            max_value=desired_distance,
            min_score=0,
            min_value=0,
            value=distance,
        )

    return reward


"""
    offset is the percetage of pixels considered as centered
"""


def calculate_pixel_reward(
    x: float,
    offset: float,
) -> float:
    center_point = 0.5
    reward = 0.0

    if value_in_range(
        value=x,
        center=center_point,
        offset=offset,
    ):
        return 3.0

    if x > center_point:
        reward = calculate_linear_score(
            max_score=1,
            max_value=center_point,
            min_score=0,
            min_value=0,
            value=x,
        )
    else:
        reward = calculate_linear_score(
            max_score=1,
            max_value=center_point,
            min_score=0,
            min_value=1,
            value=x,
        )

    return reward


def calculate_linear_score(max_score, min_score, max_value, min_value, value) -> float:
    m = (max_score - min_score) / (max_value - min_value)
    b = max_score + (m * max_value)
    score = value * m + b
    return score
