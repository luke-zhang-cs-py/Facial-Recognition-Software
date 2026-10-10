"""tools/pad_eval.py: APCER and BPCER (ISO/IEC 30107-3) from attempt results.

Synthetic rows only. The real CSV comes from consenting participants (the
protocol in notes/BENCHMARK.md) and holds outcomes, never images.
"""
import io
import json

import pytest

from tools import pad_eval

HEADER = "attack_type,ground_truth,verdict\n"


def rows(text):
    return pad_eval.read_rows(io.StringIO(HEADER + text))


def test_apcer_is_per_species_and_reported_at_the_worst_one():
    result = pad_eval.evaluate(rows(
        "print,attack,spoof\n" * 9 + "print,attack,live\n"       # 1 of 10
        + "replay,attack,spoof\n" * 2 + "replay,attack,live\n" * 2  # 2 of 4
    ))
    assert result["apcer"]["print"]["rate"] == pytest.approx(0.1)
    assert result["apcer"]["replay"]["rate"] == pytest.approx(0.5)
    worst = result["apcerWorst"]
    assert worst["species"] == "replay" and worst["errors"] == 2
    assert result["bpcer"]["n"] == 0 and result["bpcer"]["rate"] is None


def test_bpcer_counts_a_real_person_rejected_or_never_decided():
    result = pad_eval.evaluate(rows(
        "none,bona_fide,live\n" * 6 + "none,bona_fide,spoof\n"
        + ",bona_fide,unknown\n"
    ))
    bpcer = result["bpcer"]
    assert (bpcer["n"], bpcer["errors"], bpcer["unknown"]) == (8, 2, 1)
    assert bpcer["rate"] == pytest.approx(0.25)
    assert result["apcer"] == {} and result["apcerWorst"] is None


def test_an_undecided_attack_is_not_an_acceptance():
    result = pad_eval.evaluate(rows("print,attack,unknown\nprint,attack,spoof\n"))
    assert result["apcer"]["print"]["errors"] == 0
    assert result["apcer"]["print"]["unknown"] == 1


def test_values_are_read_case_and_space_insensitively_and_extra_columns_ignored():
    text = ("participant,attack_type,ground_truth,verdict\n"
            "P01, Print , ATTACK ,Live\n"
            "P02,None,Bona_Fide, LIVE\n")
    assert pad_eval.read_rows(io.StringIO(text)) == [
        ("print", "attack", "live"), ("none", "bona_fide", "live")]


@pytest.mark.parametrize("line, message", [
    ("print,attack,maybe\n", "verdict must be"),
    ("print,impostor,live\n", "ground_truth must be"),
    ("print,bona_fide,live\n", "a bona fide attempt has attack_type"),
    ("none,attack,live\n", "an attack needs its attack_type"),
])
def test_a_malformed_row_is_refused_with_its_line_number(line, message):
    with pytest.raises(ValueError) as caught:
        rows("none,bona_fide,live\n" + line)
    assert message in str(caught.value) and "line 3" in str(caught.value)


def test_a_csv_without_the_columns_is_refused():
    with pytest.raises(ValueError, match="no ground_truth, verdict column"):
        pad_eval.read_rows(io.StringIO("attack_type,result\nprint,live\n"))


def test_the_wilson_interval_is_not_zero_width_at_zero_errors():
    low, high = pad_eval.wilson(0, 20)
    assert low == 0.0 and 0.15 < high < 0.17
    assert pad_eval.wilson(0, 0) == (None, None)
    low, high = pad_eval.wilson(5, 10)
    assert low < 0.5 < high and low == pytest.approx(1 - high)


def test_main_prints_a_table_or_json_and_exits_2_on_a_bad_file(tmp_path, capsys):
    good = tmp_path / "attempts.csv"
    good.write_text(HEADER + "print,attack,live\nnone,bona_fide,unknown\n",
                    encoding="utf-8")
    assert pad_eval.main([str(good)]) == 0
    out = capsys.readouterr().out
    assert "APCER print" in out and "APCER (worst)" in out and "BPCER" in out
    assert "never reached a verdict" in out

    assert pad_eval.main([str(good), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["apcerWorst"]["rate"] == 1.0 and parsed["attempts"] == 2

    bad = tmp_path / "bad.csv"
    bad.write_text(HEADER + "print,attack,perhaps\n", encoding="utf-8")
    assert pad_eval.main([str(bad)]) == 2
    assert "verdict must be" in capsys.readouterr().err
