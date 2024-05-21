import numpy as np

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

    # Exponential continuous reward
    reward = calculate_exponential_score(
        max_score=3, value=distance, desired_value=desired_distance, k = 0.2, is_distance=True
    )

    # linear continuous reward
    # if distance > desired_distance:
    #     reward = calculate_linear_score(
    #         max_score=1,
    #         max_value=desired_distance,
    #         min_score=0,
    #         min_value=max_distace,
    #         value=distance,
    #     )
    # else:
    #     reward = calculate_linear_score(
    #         max_score=1,
    #         max_value=desired_distance,
    #         min_score=0,
    #         min_value=0,
    #         value=distance,
    #     )

    return reward


"""
    offset is the percetage of pixels considered as centered
"""


def calculate_pixel_reward(
    x: float,
    any_detected: float,
    offset: float,
    has_seen: bool
) -> float:
    center_point = 0.5
    reward = 0.0
    if not any_detected:
        return 0 if has_seen else reward

    if value_in_range(
        value=x,
        center=center_point,
        offset=offset,
    ):
        # print("PIXEL IN RANGE")
        return 3.0

    # Exponential continuous reward
    reward = calculate_exponential_score(
        max_score=2, value=x, desired_value=center_point, k=6
    )

    # Linear continuous reward
    # if x > center_point:
    #     reward = calculate_linear_score(
    #         max_score=1,
    #         max_value=center_point,
    #         min_score=0,
    #         min_value=0,
    #         value=x,
    #     )
    # else:
    #     reward = calculate_linear_score(
    #         max_score=1,
    #         max_value=center_point,
    #         min_score=0,
    #         min_value=1,
    #         value=x,
    #     )

    return reward


def calculate_linear_score(max_score, min_score, max_value, min_value, value) -> float:
    m = (max_score - min_score) / (max_value - min_value)
    b = max_score + (m * max_value)
    score = value * m + b

    return score


def calculate_exponential_score(max_score, value, desired_value, k, is_distance = False) -> float:
    delta = np.abs(value - desired_value)
    A = max_score

    if is_distance and value < desired_value:
        delta = value
        A = -3
        
    reward = A * np.exp(-k * delta)

    return reward
