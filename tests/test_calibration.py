import pytest

from nirnay import Decider, IsotonicCalibrator
from nirnay.calibration import expected_calibration_error, reliability_bins, risk_coverage


def test_isotonic_maps_vote_share_to_observed_accuracy():
    # 1.0 votes right 8/10 times, 0.5 votes right 1/2 times.
    conf = [1.0] * 10 + [0.5] * 2
    ok = [True] * 8 + [False] * 2 + [True, False]
    cal = IsotonicCalibrator().fit(conf, ok)
    assert cal(1.0) == pytest.approx(0.8)
    assert cal(0.5) == pytest.approx(0.5)
    assert cal(0.75) == pytest.approx(0.65)  # linear between knots
    assert cal(0.1) == pytest.approx(0.5)  # clamped below the lowest knot


def test_isotonic_pools_decreasing_accuracy_into_one_block():
    # 0.6 does better than 0.8 here; isotonic must not let accuracy go down as votes go up.
    conf = [0.6] * 4 + [0.8] * 4 + [1.0] * 2
    ok = [True] * 4 + [True, True, False, False] + [True, True]
    cal = IsotonicCalibrator().fit(conf, ok)
    assert cal(0.6) == pytest.approx(0.75) and cal(0.8) == pytest.approx(0.75)
    assert cal(1.0) == pytest.approx(1.0)
    values = [cal(x / 10) for x in range(11)]
    assert values == sorted(values)


def test_calibrator_round_trips_through_json(tmp_path):
    cal = IsotonicCalibrator().fit([0.5, 1.0, 1.0], [False, True, True])
    path = tmp_path / "cal.json"
    cal.save(path)
    again = IsotonicCalibrator.load(path)
    assert again.xs == cal.xs and again.ys == cal.ys and again(0.75) == cal(0.75)


def test_fit_and_use_errors():
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([], [])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.5], [True, False])
    with pytest.raises(RuntimeError):
        IsotonicCalibrator()(0.5)


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = [0.5, 0.5, 1.0, 1.0]
    assert expected_calibration_error(conf, [True, False, True, True]) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    # Always 100% sure, right half the time → ECE 0.5.
    assert expected_calibration_error([1.0] * 4, [True, False, True, False]) == pytest.approx(0.5)


def test_reliability_bins_put_one_in_the_top_bin():
    bins = reliability_bins([1.0, 0.95, 0.05], [True, False, True], n_bins=10)
    assert [(b.lo, b.count) for b in bins] == [(0.0, 1), (0.9, 2)]


def test_risk_coverage_trades_coverage_for_accuracy():
    conf = [1.0, 1.0, 0.9, 0.5]
    ok = [True, True, True, False]
    rows = risk_coverage(conf, ok, [0.0, 0.7])
    assert rows[0] == (0.0, 1.0, 0.75)
    assert rows[1] == (0.7, 0.75, 1.0)


def test_decider_applies_calibrator_and_thresholds_on_calibrated_value(fake):
    cal = IsotonicCalibrator([0.5, 1.0], [0.3, 0.8])
    fake.reply(*"BBBBBBBBBB")
    fake.reply(*"BBBBBAAAAA")
    with Decider(client=fake.client(), calibrator=cal, threshold=0.7, escalate_below=None) as d:
        sure = d.decide("Which team?", ["billing", "refund"], "refund kab milega")
        unsure = d.decide("Which team?", ["billing", "refund"], "kuch gadbad hai")
    assert sure.raw_confidence == 1.0 and sure.confidence == pytest.approx(0.8)
    assert sure.choice == "refund" and not sure.abstained
    assert unsure.raw_confidence == 0.5 and unsure.confidence == pytest.approx(0.3)
    assert unsure.abstained and unsure.choice is None


def test_calibrator_needs_multiple_samples(fake):
    with pytest.raises(ValueError):
        Decider(client=fake.client(), samples=1, calibrator=lambda x: x)
