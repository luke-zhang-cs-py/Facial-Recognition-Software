"""calibration.py is pure arithmetic over measured tables -- the easiest thing
in the project to get subtly wrong and the hardest to notice, since a wrong
threshold still returns a plausible number."""
import pytest

import calibration as c


def test_fmr_is_monotonic_in_threshold():
    rates = [c.fmr_at(t) for t, _ in c.SFACE_FMR]
    assert all(a >= b for a, b in zip(rates, rates[1:])), \
        "a stricter threshold must never raise the false match rate"


def test_fmr_interpolates_between_measured_points():
    lo, hi = c.SFACE_FMR[0][0], c.SFACE_FMR[1][0]
    mid = (lo + hi) / 2
    assert c.fmr_at(hi) <= c.fmr_at(mid) <= c.fmr_at(lo)


def test_fmr_clamps_outside_the_measured_range():
    assert c.fmr_at(-5) == c.SFACE_FMR[0][1]
    assert c.fmr_at(99) == c.SFACE_FMR[-1][1]


@pytest.mark.parametrize("n", [2, 10, 100, 1000, 10000])
def test_gallery_risk_grows_with_gallery_size(n):
    assert c.gallery_risk(0.5, n) <= c.gallery_risk(0.5, n * 2)


def test_gallery_risk_is_zero_for_a_gallery_of_one():
    assert c.gallery_risk(0.5, 1) == 0.0
    assert c.gallery_risk(0.5, 0) == 0.0


def test_recommend_threshold_tightens_as_the_gallery_grows():
    small, _, _ = c.recommend_threshold(10)
    large, _, _ = c.recommend_threshold(1000)
    assert large >= small


def test_recommend_threshold_reports_when_the_target_is_unreachable():
    _, _, ok = c.recommend_threshold(10)
    assert ok
    _, risk, ok_big = c.recommend_threshold(100000)
    assert not ok_big and risk > 0.01, \
        "must admit it cannot hit the target rather than returning a number anyway"


def test_age_coverage_is_monotonic():
    widths = sorted(c.AGE_COVERAGE)
    cov = [c.age_band_coverage(w) for w in widths]
    assert all(a <= b for a, b in zip(cov, cov[1:])), \
        "a wider band must never contain the truth less often"


def test_age_band_is_honest_about_being_narrow():
    assert c.age_band_coverage(2) < 0.2, \
        "the +/-2y band is ~10% accurate; if this passes 20% the table is wrong"


def test_describe_age_states_the_coverage():
    text = c.describe_age(30)
    assert "%" in text and "30" in text


def test_sample_accuracy_saturates():
    at_sat = c.accuracy_for_samples(c.SAMPLE_SATURATION)
    assert c.accuracy_for_samples(c.SAMPLE_SATURATION * 2) == pytest.approx(at_sat)
    assert c.accuracy_for_samples(1) < at_sat


def test_sample_accuracy_clamps():
    assert c.accuracy_for_samples(0) == c.SAMPLE_ACCURACY[min(c.SAMPLE_ACCURACY)]
    assert c.accuracy_for_samples(9999) == c.SAMPLE_ACCURACY[max(c.SAMPLE_ACCURACY)]
