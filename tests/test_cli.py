"""Command-line input validation (fails before any data is loaded)."""
import pytest

from main import build_parser, main


@pytest.mark.parametrize("user", ["0", "944", "-5"])
def test_user_out_of_range_is_rejected(user):
    with pytest.raises(SystemExit, match="between 1 and 943"):
        main(["recommend", "--user", user, "--method", "nb"])


@pytest.mark.parametrize("n", ["0", "-3"])
def test_n_must_be_positive(n):
    with pytest.raises(SystemExit, match="-n must be positive"):
        main(["recommend", "--user", "196", "--method", "nb", "-n", n])


@pytest.mark.parametrize("args", [["recommend", "--user", "1", "--method", "xx"],
                                  ["recommend", "--user", "1", "--method", "nb", "--fold", "6"],
                                  ["recommend", "--user", "abc", "--method", "nb"]])
def test_argparse_rejects_bad_choices(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_all_commands_are_registered():
    parser = build_parser()
    for command in (["evaluate"], ["experiments", "--fresh"], ["plots"], ["human-eval", "--users", "3"]):
        assert parser.parse_args(command).command == command[0]
