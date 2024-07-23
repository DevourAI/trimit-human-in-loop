from trimit.utils.namegen import timeline_namegen, project_namegen


def test_namegen():
    assert timeline_namegen().startswith("timeline-")
    assert project_namegen().startswith("project-")
    assert timeline_namegen() != timeline_namegen()
    assert project_namegen() != project_namegen()
