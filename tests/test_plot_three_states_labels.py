import os
from speed.plot_three_states import plot_three_states


def test_plot_three_states_saves_and_returns_segments(tmp_path):
    input_csv = os.path.join(os.path.dirname(__file__), '..', 'speed', 'speed.csv')
    out_file = tmp_path / 'three_states_test.png'
    summary = tmp_path / 'three_states_test.csv'

    out, segments = plot_three_states(input_csv=input_csv, out=str(out_file), show=False, summary_csv=str(summary))
    # file saved and returned path should match
    assert os.path.exists(out)
    assert out == str(out_file)
    # segments should be a list of dicts with start/end/label
    assert isinstance(segments, list)
    assert all(isinstance(s, dict) for s in segments)
    for s in segments:
        assert 'label' in s and 'start' in s and 'end' in s
        assert float(s['start']) <= float(s['end'])

    assert os.path.exists(str(summary))
