from analytics.latex_utils import clean_moodle_latex, split_stack_debug_dump


def test_split_stack_debug_dump_isolates_a_leaked_maxima_variable_trace():
    """Reproduces the reported symptom shape: real question prose followed by a chain of
    unsuppressed Maxima statement echoes (STACK's CAS session leaking through after a
    runtime error on this random seed). Deliberately uses different variable names than
    the bug report to confirm the fix isn't overfit to that one question's variables."""
    text = (
        r"Add the complex numbers \(3+4\mathrm{i}\) and \(1-2\mathrm{i}\) in polar form."
        r" theta_tot = 0.927 ; r_tot = 5.385 ; thetas = [0.927,-1.107] ; rs = [5.0,2.236] ;"
        r" trigs = [0.6,0.8,0.447,-0.894] ; trig_tot = [3.578,3.753] ; rects = [3.578,3.753]"
    )
    clean, dump = split_stack_debug_dump(text)
    assert clean == r"Add the complex numbers \(3+4\mathrm{i}\) and \(1-2\mathrm{i}\) in polar form."
    assert "theta_tot" in dump
    assert "rects" in dump


def test_split_stack_debug_dump_leaves_a_single_inline_equation_alone():
    """A lone `x = 5` inside real question prose is completely ordinary and must not be
    mistaken for a leaked dump — only a *chain* of 2+ assignments is the CAS signature."""
    text = r"Solve for \(x\) where \(x = 5\) and \(y = 2x + 1\)."
    clean, dump = split_stack_debug_dump(text)
    assert clean == text
    assert dump == ""


def test_split_stack_debug_dump_leaves_boolean_prt_outcomes_alone():
    r"""`\mathbf{True}`/`\mathbf{False}` is ordinary STACK rendering of a boolean PRT
    outcome and must render normally, not be swept into the debug dump."""
    text = r"Is the statement \(\mathbf{True}\) or \(\mathbf{False}\)? Evaluate \(2+2=4\)."
    clean, dump = split_stack_debug_dump(text)
    assert clean == text
    assert dump == ""


def test_split_stack_debug_dump_preserves_multiple_real_sub_expressions():
    """Several genuine `\\(...\\)` sub-expressions ahead of a trailing dump must all
    survive; only the leaked chain is split off the end."""
    text = (
        r"Given \(a=3\) find \(\sqrt{a}\) and \(a^2\)."
        r" scratch1 = 9; scratch2 = 3.0; scratch3 = 27"
    )
    clean, dump = split_stack_debug_dump(text)
    assert clean == r"Given \(a=3\) find \(\sqrt{a}\) and \(a^2\)."
    assert dump == "scratch1 = 9; scratch2 = 3.0; scratch3 = 27"


def test_split_stack_debug_dump_handles_empty_and_clean_text():
    assert split_stack_debug_dump("") == ("", "")
    clean_only = r"What is \(\sqrt{16}\)?"
    assert split_stack_debug_dump(clean_only) == (clean_only, "")


def test_clean_moodle_latex_still_renders_the_isolated_prompt():
    """End-to-end: split off the dump, then run the remainder through the normal
    delimiter converter, exactly as the page does — confirms the two functions compose."""
    text = (
        r"Find \(\sqrt{2}\) to 3 d.p. theta_tot = 1; r_tot = 2; thetas = [1]; rs = [2];"
    )
    clean, dump = split_stack_debug_dump(text)
    rendered = clean_moodle_latex(clean)
    assert rendered == r"Find $\sqrt{2}$ to 3 d.p."
    assert dump.startswith("theta_tot")
