"""Unit tests for eval/grader.py — answer extraction, normalization, and comparison."""

import sys
from pathlib import Path

import pytest

# Ensure eval/ is on sys.path so we can import grader directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import (
    extract_raw_boxed,
    _strip_latex_cruft,
    _fix_fracs,
    _fix_a_slash_b,
    _fix_sqrt,
    strip_string,
    parse_digits,
    is_digit,
    str_to_pmatrix,
    numeric_equal,
    symbolic_equal,
    math_equal,
)


# ============================================================
# extract_raw_boxed
# ============================================================
class TestExtractRawBoxed:
    def test_simple_boxed(self):
        assert extract_raw_boxed(r"The answer is $\boxed{42}$.") == "42"

    def test_latex_frac(self):
        assert extract_raw_boxed(r"$\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"

    def test_nested_braces(self):
        assert extract_raw_boxed(r"\boxed{\frac{a}{b+{c}}}") == r"\frac{a}{b+{c}}"

    def test_multiple_boxed_takes_last(self):
        text = r"\boxed{1} and then \boxed{2}"
        assert extract_raw_boxed(text) == "2"

    def test_no_boxed_returns_none(self):
        assert extract_raw_boxed("No boxed here") is None

    def test_empty_boxed(self):
        assert extract_raw_boxed(r"\boxed{}") == ""

    def test_strips_text_command(self):
        assert extract_raw_boxed(r"\boxed{\text{yes}}") == "yes"

    def test_strips_circ(self):
        assert extract_raw_boxed(r"\boxed{90^\circ}") == "90"

    def test_non_string_input(self):
        assert extract_raw_boxed(123) is None  # no \boxed in "123"

    def test_boxed_with_spaces(self):
        assert extract_raw_boxed(r"\boxed  {7}") == "7"


# ============================================================
# _strip_latex_cruft
# ============================================================
class TestStripLatexCruft:
    def test_removes_text_command(self):
        assert _strip_latex_cruft(r"\text{hello}") == "hello"

    def test_removes_circ(self):
        assert _strip_latex_cruft(r"90^\circ") == "90"
        assert _strip_latex_cruft(r"90\circ") == "90"

    def test_strips_whitespace(self):
        assert _strip_latex_cruft("  42  ") == "42"


# ============================================================
# _fix_fracs
# ============================================================
class TestFixFracs:
    def test_already_correct(self):
        assert _fix_fracs(r"\frac{1}{2}") == r"\frac{1}{2}"

    def test_shorthand_frac(self):
        # \frac12 -> \frac{1}{2}
        assert _fix_fracs(r"\frac12") == r"\frac{1}{2}"

    def test_shorthand_frac_with_trailing(self):
        assert _fix_fracs(r"\frac12 + x") == r"\frac{1}{2} + x"

    def test_one_brace_missing(self):
        # \frac1{2} -> \frac{1}{2}
        assert _fix_fracs(r"\frac1{2}") == r"\frac{1}{2}"

    def test_no_frac(self):
        assert _fix_fracs("x + y") == "x + y"


# ============================================================
# _fix_a_slash_b
# ============================================================
class TestFixASlashB:
    def test_simple_fraction(self):
        assert _fix_a_slash_b("3/4") == r"\frac{3}{4}"

    def test_not_a_fraction(self):
        assert _fix_a_slash_b("hello") == "hello"

    def test_multiple_slashes(self):
        assert _fix_a_slash_b("1/2/3") == "1/2/3"  # returns as-is

    def test_non_integer_parts(self):
        assert _fix_a_slash_b("a/b") == "a/b"  # can't parse as int

    def test_sqrt_passthrough(self):
        # sqrt in numerator should keep it
        assert _fix_a_slash_b("sqrt2/3") is not None  # shouldn't crash


# ============================================================
# _fix_sqrt
# ============================================================
class TestFixSqrt:
    def test_adds_braces(self):
        assert _fix_sqrt(r"\sqrt2") == r"\sqrt{2}"

    def test_already_braced(self):
        assert _fix_sqrt(r"\sqrt{2}") == r"\sqrt{2}"

    def test_word_after_sqrt(self):
        assert _fix_sqrt(r"\sqrtx") == r"\sqrt{x}"


# ============================================================
# strip_string
# ============================================================
class TestStripString:
    def test_strips_whitespace(self):
        assert strip_string("  42  ") == "42"

    def test_removes_trailing_dot(self):
        assert strip_string("42.") == "42"

    def test_removes_newlines(self):
        assert strip_string("4\n2") == "42"

    def test_tfrac_to_frac(self):
        assert strip_string(r"\tfrac{1}{2}") == r"\frac{1}{2}"

    def test_dfrac_to_frac(self):
        assert strip_string(r"\dfrac{1}{2}") == r"\frac{1}{2}"

    def test_removes_left_right(self):
        assert strip_string(r"\left(\frac{1}{2}\right)") == r"(\frac{1}{2})"

    def test_removes_dollar_signs(self):
        assert strip_string("$42$") == "42"

    def test_removes_percent(self):
        assert strip_string(r"50\%") == "50"
        assert strip_string("50%") == "50"

    def test_leading_zero(self):
        assert strip_string(".5") == "0.5"

    def test_trailing_decimal_zeros(self):
        assert strip_string("3.000") == "3"
        assert strip_string("3.00x") == "3x"

    def test_strips_equation_prefix(self):
        assert strip_string("x = 42") == "42"

    def test_matrix_normalization(self):
        s = r"\begin{bmatrix}1\\2\end{bmatrix}"
        result = strip_string(s)
        assert "pmatrix" in result

    def test_removes_inverse_space(self):
        assert strip_string(r"a\!b") == "ab"

    def test_infinity(self):
        assert strip_string("infinity") == r"\infty"

    def test_empty_string(self):
        assert strip_string("") == ""

    def test_removes_spaces(self):
        assert strip_string("1 + 2") == "1+2"

    def test_slash_to_frac(self):
        assert strip_string("3/4") == r"\frac{3}{4}"


# ============================================================
# parse_digits / is_digit
# ============================================================
class TestParseDigits:
    def test_integer(self):
        assert parse_digits("42") == 42.0

    def test_float(self):
        assert parse_digits("3.14") == pytest.approx(3.14)

    def test_with_commas(self):
        assert parse_digits("1,000") == 1000.0

    def test_percentage(self):
        assert parse_digits("50%") == pytest.approx(0.5)

    def test_percentage_with_backslash(self):
        assert parse_digits("50\\%") == pytest.approx(0.5)

    def test_non_number(self):
        assert parse_digits("abc") is None

    def test_negative(self):
        assert parse_digits("-7") == -7.0


class TestIsDigit:
    def test_number(self):
        assert is_digit("42") is True

    def test_not_number(self):
        assert is_digit("abc") is False

    def test_percentage(self):
        assert is_digit("50%") is True


# ============================================================
# str_to_pmatrix
# ============================================================
class TestStrToPmatrix:
    def test_single_matrix(self):
        result = str_to_pmatrix("{1,2}")
        assert r"\begin{pmatrix}" in result
        assert r"\end{pmatrix}" in result

    def test_no_matrix(self):
        result = str_to_pmatrix("42")
        assert result == ""


# ============================================================
# numeric_equal
# ============================================================
class TestNumericEqual:
    def test_equal(self):
        assert numeric_equal(1.0, 1.0) is True

    def test_close(self):
        assert numeric_equal(1.0, 1.00005) is True

    def test_not_close(self):
        assert numeric_equal(1.0, 2.0) is False

    def test_zero(self):
        assert numeric_equal(0.0, 0.0) is True


# ============================================================
# symbolic_equal
# ============================================================
class TestSymbolicEqual:
    def test_identical_strings(self):
        assert symbolic_equal("x + 1", "x + 1") is True

    def test_equivalent_expressions(self):
        assert symbolic_equal("x + 1", "1 + x") is True

    def test_different_expressions(self):
        assert symbolic_equal("x + 1", "x + 2") is False

    def test_latex_fractions(self):
        assert symbolic_equal(r"\frac{1}{2}", r"\frac{2}{4}") is True

    def test_numeric_values(self):
        assert symbolic_equal("0.5", r"\frac{1}{2}") is True


# ============================================================
# math_equal — 5-tier comparison
# ============================================================
class TestMathEqual:
    # Tier 1: exact string match
    def test_exact_match(self):
        assert math_equal("42", "42") is True

    def test_case_insensitive(self):
        assert math_equal("True", "true") is True

    # Tier 2: numeric equality
    def test_numeric_equal(self):
        assert math_equal("0.5", "0.500") is True

    def test_numeric_with_commas(self):
        assert math_equal("1000", "1,000") is True

    def test_percentage_variant(self):
        assert math_equal("50", "0.5", include_percentage=True) is True

    def test_no_percentage(self):
        assert math_equal("50", "0.5", include_percentage=False) is False

    # None handling
    def test_none_prediction(self):
        assert math_equal(None, "42") is False

    def test_none_reference(self):
        assert math_equal("42", None) is False

    # Tier 3: format normalization
    def test_brackets_stripped(self):
        assert math_equal("[42]", "42") is True

    def test_parens_stripped(self):
        assert math_equal("(42)", "42") is True

    def test_list_comparison(self):
        assert math_equal("[1, 2]", "[1, 2]") is True

    def test_list_different(self):
        assert math_equal("[1, 2]", "[1, 3]") is False

    # Equation form
    def test_equation_prefix_in_pred(self):
        assert math_equal("x = 42", "42") is True

    def test_equation_prefix_in_ref(self):
        assert math_equal("42", "x = 42") is True

    # Tier 4: symbolic equality
    def test_symbolic_frac(self):
        assert math_equal(r"\frac{1}{2}", r"\frac{2}{4}") is True

    def test_symbolic_different(self):
        assert math_equal(r"\frac{1}{2}", r"\frac{1}{3}") is False

    # Simple wrong answers
    def test_wrong_number(self):
        assert math_equal("5", "7") is False

    def test_empty_prediction(self):
        assert math_equal("", "42") is False

    # Matrix comparison
    def test_pmatrix_equal(self):
        pred = r"\begin{pmatrix}1\\2\end{pmatrix}"
        ref = r"\begin{pmatrix}1\\2\end{pmatrix}"
        assert math_equal(pred, ref) is True

    def test_pmatrix_different(self):
        pred = r"\begin{pmatrix}1\\2\end{pmatrix}"
        ref = r"\begin{pmatrix}1\\3\end{pmatrix}"
        assert math_equal(pred, ref) is False
