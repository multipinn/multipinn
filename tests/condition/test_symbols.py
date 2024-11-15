import pytest

from multipinn.condition.symbols import Symbols


def test_symbols_initialization():
    symbols = Symbols(variables="x, y, t", functions="u, v")
    assert symbols.variables == ["x", "y", "t"]
    assert symbols.functions == ["u", "v"]
    assert symbols.input_dim == 3
    assert symbols.output_dim == 2


def test_check_names_valid():
    symbols = Symbols(variables="x, y, t", functions="u, v")
    # This should not raise an exception
    symbols.check_names()


def test_check_names_invalid():
    with pytest.raises(ValueError, match="Names must be a single letter"):
        Symbols(variables="x, y, t", functions="u, velocity")

    with pytest.raises(ValueError, match="Duplicate names"):
        Symbols(variables="x, y, x", functions="u, v")


def test_check_format_valid():
    symbols = Symbols(variables="x, y", functions="u")
    # These should not raise exceptions
    symbols.check_format(["x", "y", "u", "u_x", "u_xy"])


def test_check_format_invalid():
    symbols = Symbols(variables="x, y", functions="u")
    with pytest.raises(
        ValueError, match="Symbol 'z' is not a defined function or variable"
    ):
        symbols.check_format(["z"])

    with pytest.raises(ValueError, match="Invalid symbol format: 'u__x'"):
        symbols.check_format(["u__x"])

    with pytest.raises(ValueError, match="Variable part cannot be empty"):
        symbols.check_format(["u_"])

    with pytest.raises(ValueError, match="Function 'v' is not defined"):
        symbols.check_format(["v_x"])

    with pytest.raises(ValueError, match="Variable 'xz' contains undefined variables"):
        symbols.check_format(["u_xz"])


def test_parsing():
    symbols = Symbols(variables="x, y", functions="u")
    assert symbols.parsing("x, y, u, u_x, u_xy") == ["x", "y", "u", "u_x", "u_xy"]

    with pytest.raises(ValueError, match="No reason to call this function"):
        symbols.parsing("")


def test_generate_str():
    symbols = Symbols(variables="x, y", functions="u")
    code = symbols.generate_str(["x", "y", "u", "u_x", "u_xy"])
    assert "def compute_gradients(model, arg):" in code
    assert "(u,) = unpack(model(arg))" in code
    assert "x, y = unpack(arg)" in code
    assert "u_x, u_y = unpack(grad(u, arg))" in code
    assert "u_xx, u_xy = unpack(grad(u_x, arg))" in code
    assert "return x, y, u, u_x, u_xy" in code


def test_call():
    symbols = Symbols(variables="x, y", functions="u")
    compute_gradients = symbols("x, y, u, u_x, u_xy")
    assert callable(compute_gradients)
    # Note: We can't easily test the actual function behavior here without mocking torch and grad
