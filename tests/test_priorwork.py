"""Citation checking: was a cited paper really in the search results the agent was shown?"""

import pytest

import priorwork

SHOWN = [
    {"url": "http://arxiv.org/abs/1803.04526v2", "title": "Kepler Data Validation I"},
    {"url": "http://arxiv.org/abs/astro-ph/0206228v1", "title": "An older paper"},
]


@pytest.mark.parametrize(
    "citation",
    [
        "https://arxiv.org/abs/1803.04526",
        "https://arxiv.org/pdf/1803.04526",
        "arxiv.org/abs/1803.04526.pdf",
        "arXiv:1803.04526",
        "http://arxiv.org/abs/1803.04526v1",
        "https://arxiv.org/abs/astro-ph/0206228",
    ],
)
def test_every_form_of_a_shown_paper_counts_as_from_the_search(citation):
    assert priorwork.check_citations([citation], SHOWN)["from_search"] == 1


def test_a_paper_that_was_not_shown_counts_as_from_memory():
    result = priorwork.check_citations(["https://arxiv.org/abs/1201.5424"], SHOWN)
    assert result == {"total": 1, "from_search": 0, "from_memory": ["https://arxiv.org/abs/1201.5424"]}


def test_a_single_string_is_one_citation_not_one_per_character():
    assert priorwork.check_citations("https://arxiv.org/abs/1803.04526", SHOWN)["total"] == 1


def test_citations_given_as_objects_are_read_by_their_url():
    cited = [{"title": "Kepler DV", "url": "https://arxiv.org/abs/1803.04526"}]
    assert priorwork.check_citations(cited, SHOWN)["from_search"] == 1


def test_nothing_cited_is_nothing_counted():
    assert priorwork.check_citations(None, SHOWN) == {"total": 0, "from_search": 0, "from_memory": []}


def test_duplicate_papers_are_merged():
    papers = [{"url": "a", "title": "x"}, {"url": "a", "title": "x"}, {"url": "b", "title": "y"}]
    assert [p["url"] for p in priorwork._dedupe(papers)] == ["a", "b"]


def test_the_prompt_lists_only_the_papers_that_are_shown():
    papers = [
        {"url": f"http://arxiv.org/abs/2601.0000{i}", "title": f"Paper {i}", "published": "2026"}
        for i in range(9)
    ]
    text = priorwork.for_prompt(papers)
    assert text.count("\n") + 1 == priorwork.SHOWN
