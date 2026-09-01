"""what the live map will draw"""
from src.serve.correction import LEAD_BUCKET_MAX_MIN
from src.serve.live_map import _compute_train_positions


class _Schedule:
    def get_leg(self, trip_id, stop_id):
        return {
            "prev_latlon": (42.3954, -71.1425),
            "next_latlon": (42.3967, -71.1218),
            "next_stop_name": "Davis",
            "leg_duration_sec": 180.0,
            "shape_id": "S1",
            "headsign": "Braintree",
            "direction_id": 0,
        }


class _Shapes:
    def position_along(self, shape_id, prev_latlon, next_latlon, frac):
        return [42.396, -71.132]


def _pred(lead_sec):
    return {"stop_id": "70063", "route_id": "Red", "lead_sec": lead_sec}


def _lookup():
    import pandas as pd
    empty = pd.DataFrame(columns=["route_id", "stop_id", "lead_bucket", "bias_sec", "n"])
    return {"by_stop": empty, "by_route": empty, "by_bucket": empty}


def test_trains_beyond_the_top_lead_bucket_are_not_drawn():
    """a trip an hour out has not departed, so there is no honest position or correction for it"""
    over = (LEAD_BUCKET_MAX_MIN + 1) * 60
    out = _compute_train_positions({"T1": _pred(over)}, _Schedule(), _Shapes(), _lookup())
    assert out == []


def test_trains_inside_the_top_lead_bucket_are_drawn():
    under = (LEAD_BUCKET_MAX_MIN - 1) * 60
    out = _compute_train_positions({"T1": _pred(under)}, _Schedule(), _Shapes(), _lookup())
    assert len(out) == 1
    assert out[0]["next_stop_name"] == "Davis"
