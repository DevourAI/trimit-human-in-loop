from trimit.backend.models import TranscriptSegment, Transcript, OffsetToCut, Soundbites
import pytest

from trimit.backend.utils import linearize_and_dedupe_offsets

pytestmark = pytest.mark.asyncio(scope="session")


async def test_transcript_segment(raw_transcript):
    segment = TranscriptSegment.from_video_transcription_segment(
        raw_transcript["segments"][0]
    )

    def check_original_segment(segment):
        assert segment.words == ["Yeah,", "sorry,", "what", "is", "his", "number?"]
        assert segment.kept_words == segment.words
        assert segment.text == "Yeah, sorry, what is his number?"
        assert segment.speaker == "SPEAKER_04"
        assert segment.start == 1.238
        assert segment.end == 2.299
        assert segment.word_start_ends == [
            (1.238, 1.478),
            (1.498, 1.718),
            (1.738, 1.858),
            (1.898, 1.938),
            (1.998, 2.058),
            (2.098, 2.299),
        ]
        assert segment.cut_segments == [(0, 6)]
        for i in range(len(segment.words)):
            assert segment.word_i_kept(i)

    check_original_segment(segment)

    segment_copy = segment.copy()
    segment_copy.cut_segments = []
    segment_copy.cut(2, 4)
    assert segment_copy.cut_segments == [(2, 4)]
    assert segment_copy.words == ["Yeah,", "sorry,", "what", "is", "his", "number?"]
    assert segment_copy.kept_words == ["what", "is"]
    assert segment_copy.text == "what is"
    assert all([not segment_copy.word_i_kept(i) for i in [0, 1, 4, 5]])
    assert segment_copy.word_i_kept(2)
    assert segment_copy.word_i_kept(3)
    check_original_segment(segment)

    with pytest.raises(ValueError):
        segment_copy.cut(2, 4)
    with pytest.raises(ValueError):
        segment_copy.cut(1, 4)
    with pytest.raises(ValueError):
        segment_copy.cut(1, 3)
    with pytest.raises(ValueError):
        segment_copy.cut(6, 7)
    with pytest.raises(ValueError):
        segment_copy.cut(3, 7)

    segment_copy.cut(1, 2)
    assert segment_copy.words == ["Yeah,", "sorry,", "what", "is", "his", "number?"]
    assert segment_copy.kept_words == ["sorry,", "what", "is"]
    assert segment_copy.text == "sorry, what is"
    assert segment_copy.cut_segments == [(1, 2), (2, 4)]
    assert all([not segment_copy.word_i_kept(i) for i in [0, 4, 5]])
    assert all([segment_copy.word_i_kept(i) for i in [1, 2, 3]])
    check_original_segment(segment)

    segment_copy.cut(5, 6)
    assert segment_copy.words == ["Yeah,", "sorry,", "what", "is", "his", "number?"]
    assert segment_copy.kept_words == ["sorry,", "what", "is", "number?"]
    assert segment_copy.text == "sorry, what is number?"
    assert segment_copy.cut_segments == [(1, 2), (2, 4), (5, 6)]
    assert all([not segment_copy.word_i_kept(i) for i in [0, 4]])
    assert all([segment_copy.word_i_kept(i) for i in [1, 2, 3, 5]])
    check_original_segment(segment)

    segment_copy_2 = segment_copy.copy()
    left, right = segment_copy_2.split(0)
    assert left.words == []
    assert left.kept_words == []
    assert left.cut_segments == []
    assert right.words == ["Yeah,", "sorry,", "what", "is", "his", "number?"]
    assert right.kept_words == ["sorry,", "what", "is", "number?"]
    assert right.cut_segments == [(1, 2), (2, 4), (5, 6)]

    segment_copy_2 = segment_copy.copy()
    left, right = segment_copy_2.split(1)
    assert left.words == ["Yeah,"]
    assert left.kept_words == []
    assert left.cut_segments == []
    assert right.cut_segments == [(0, 1), (1, 3), (4, 5)]
    assert right.words == ["sorry,", "what", "is", "his", "number?"]
    assert right.kept_words == ["sorry,", "what", "is", "number?"]

    segment_copy_2 = segment_copy.copy()
    left, right = segment_copy_2.split(2)
    assert left.words == ["Yeah,", "sorry,"]
    assert left.kept_words == ["sorry,"]
    assert left.cut_segments == [(1, 2)]
    assert right.words == ["what", "is", "his", "number?"]
    assert right.kept_words == ["what", "is", "number?"]
    assert right.cut_segments == [(0, 2), (3, 4)]

    segment_copy_2 = segment_copy.copy()
    left, right = segment_copy_2.split(3)
    assert left.words == ["Yeah,", "sorry,", "what"]
    assert left.kept_words == ["sorry,", "what"]
    assert left.cut_segments == [(1, 2), (2, 3)]
    assert right.words == ["is", "his", "number?"]
    assert right.kept_words == ["is", "number?"]
    assert right.cut_segments == [(0, 1), (2, 3)]

    segment_copy_2 = segment_copy.copy()
    left, right = segment_copy_2.split(5)
    assert left.words == ["Yeah,", "sorry,", "what", "is", "his"]
    assert left.kept_words == ["sorry,", "what", "is"]
    assert left.cut_segments == [(1, 2), (2, 4)]
    assert right.words == ["number?"]
    assert right.kept_words == ["number?"]
    assert right.cut_segments == [(0, 1)]

    segment_copy_2 = segment_copy.copy()
    with pytest.raises(ValueError):
        left, right = segment_copy_2.split(6)


async def test_transcript_split_in_chunks(raw_transcript):
    transcript = Transcript.from_video_transcription(raw_transcript)
    assert len(transcript.segments) == 9
    transcript_copy = transcript.copy()
    chunks = transcript_copy.split_in_chunks(6)
    assert [len(s.words) > 0 for s in transcript_copy.segments]
    assert chunks[0].chunk_segment_indexes == [0]
    assert chunks[0].text == "Yeah, sorry, what is his number?"
    assert chunks[1].chunk_segment_indexes == [1, 2]
    assert chunks[1].text == "My case number. OK, yeah."
    assert chunks[2].chunk_segment_indexes == [3, 4]
    assert chunks[2].text == "Oh, fuck. Your case number."
    assert chunks[3].chunk_segment_indexes == [5, 6, 7]
    assert chunks[3].text == "Sorry. Sorry, I'm not something. Sorry?"
    assert chunks[4].chunk_segment_indexes == [8]
    assert chunks[4].text == "Yeah, I'm ready, yeah."
    assert transcript_copy.kept_words == [
        w for chunk in chunks for w in chunk.kept_words
    ]

    assert chunks[1].start_offset == 6
    assert chunks[2].start_offset == 11
    chunk_2_copy = chunks[2].copy()
    chunk_2_copy.segments[1].cut_segments = []
    chunk_2_copy.segments[1].cut(1, 2)
    chunks[2].erase_cuts()
    assert chunks[2].segments[1].cut_segments == []
    assert chunk_2_copy.segments[1].cut_segments == [(1, 2)]

    chunk_3_copy = chunks[3].copy()
    chunk_3_copy.erase_cuts()
    chunk_3_copy.transcript.kept_segments.add(6)
    chunk_3_copy.transcript.kept_segments.add(7)
    chunk_3_copy.segments[1].cut(1, 3)
    full_offsets = chunk_2_copy.full_offsets_from_kept_offsets([(0, 1)])
    full_offsets_2 = chunk_3_copy.full_offsets_from_kept_offsets([(0, 1), (1, 2)])
    assert full_offsets == [
        OffsetToCut(seg_i_start=3, word_i_start=0, seg_i_end=3, word_i_end=1)
    ]
    assert full_offsets_2 == [
        OffsetToCut(seg_i_start=6, word_i_start=1, seg_i_end=6, word_i_end=2),
        OffsetToCut(seg_i_start=6, word_i_start=2, seg_i_end=6, word_i_end=3),
    ]

    assert transcript_copy.chunks[2].segments[1].cut_segments == []
    transcript_copy_2 = transcript_copy.copy()
    transcript_copy.keep_only_cuts(full_offsets, from_chunk=chunk_2_copy)
    transcript_copy_2.keep_only_cuts(full_offsets + full_offsets_2)
    assert transcript_copy.chunks[2].text == "Oh,"
    assert transcript_copy_2.chunks[1].text == ""
    assert transcript_copy_2.chunks[2].text == "Oh,"
    assert transcript_copy_2.chunks[3].text == "I'm not"

    full_offsets = (
        chunk_3_copy.transcript.contiguous_full_word_offsets_from_kept_offset((12, 16))
    )
    assert full_offsets == [
        OffsetToCut(seg_i_start=6, word_i_start=2, seg_i_end=6, word_i_end=3),
        OffsetToCut(seg_i_start=8, word_i_start=0, seg_i_end=8, word_i_end=3),
    ]


async def test_words_with_keep_tags(raw_transcript):
    transcript = Transcript.from_video_transcription(raw_transcript)
    segment = transcript.segments[0]
    assert segment.words_with_keep_tags == [
        "<keep>",
        "Yeah,",
        "sorry,",
        "what",
        "is",
        "his",
        "number?",
        "</keep>",
    ]
    segment.cut(2, 4)
    assert segment.words_with_keep_tags == [
        "Yeah,",
        "sorry,",
        "<keep>",
        "what",
        "is",
        "</keep>",
        "his",
        "number?",
    ]
    assert (
        transcript.text_with_keep_tags
        == "Yeah, sorry, <keep> what is </keep> his number? <keep> My case number. </keep> <keep> OK, yeah. </keep> <keep> Oh, fuck. </keep> <keep> Your case number. </keep> <keep> Sorry. </keep> <keep> Sorry, I'm not something. </keep> <keep> Sorry? </keep> <keep> Yeah, I'm ready, yeah. </keep>"
    )
    chunks = transcript.split_in_chunks(10)
    chunks[1].kept_segments[2].cut(1, 3)
    assert (
        chunks[1].text_with_keep_tags
        == "<keep> OK, yeah. </keep> <keep> Oh, fuck. </keep> Your <keep> case number. </keep> <keep> Sorry. </keep>"
    )


async def test_words_with_keep_and_remove_tags(raw_transcript):
    transcript = Transcript.from_video_transcription(raw_transcript)
    segment = transcript.segments[0]
    assert segment.words_with_keep_and_remove_tags == [
        "<keep>",
        "Yeah,",
        "sorry,",
        "what",
        "is",
        "his",
        "number?",
        "</keep>",
    ]
    segment.cut(2, 4)
    assert segment.words_with_keep_and_remove_tags == [
        "<remove>",
        "Yeah,",
        "sorry,",
        "</remove>",
        "<keep>",
        "what",
        "is",
        "</keep>",
        "<remove>",
        "his",
        "number?",
        "</remove>",
    ]
    assert (
        transcript.text_with_keep_and_remove_tags
        == "<remove> Yeah, sorry, </remove> <keep> what is </keep> <remove> his number? </remove> <keep> My case number. </keep> <keep> OK, yeah. </keep> <keep> Oh, fuck. </keep> <keep> Your case number. </keep> <keep> Sorry. </keep> <keep> Sorry, I'm not something. </keep> <keep> Sorry? </keep> <keep> Yeah, I'm ready, yeah. </keep>"
    )
    chunks = transcript.split_in_chunks(10)
    chunks[1].kept_segments[2].cut(1, 3)
    chunks[1].kept_segments[0].cut_segments = []
    assert (
        chunks[1].text_with_keep_and_remove_tags
        == "<remove> OK, yeah. </remove> <keep> Oh, fuck. </keep> <remove> Your </remove> <keep> case number. </keep> <keep> Sorry. </keep>"
    )


async def test_soundbites_merge(
    soundbites_chunk_1_3909774043,
    soundbites_chunk_2_3909774043,
    soundbites_chunk_3_3909774043,
):
    breakpoint()
    soundbites = Soundbites.merge(
        soundbites_chunk_1_3909774043,
        soundbites_chunk_2_3909774043,
        soundbites_chunk_3_3909774043,
    )
    assert len(soundbites.soundbites) == 6
    assert soundbites.soundbites[0:2] == soundbites_chunk_1_3909774043.soundbites
    assert soundbites.soundbites[2:4] == soundbites_chunk_2_3909774043.soundbites
    assert soundbites.soundbites[4:6] == soundbites_chunk_3_3909774043.soundbites


async def test_soundbites_iter_text(soundbites_3909774043):
    iter_text_list = [(i, text) for i, text in soundbites_3909774043.iter_text()]
    assert iter_text_list[0][1] == "Surkana and LiveRamp"
    assert (
        iter_text_list[-1][1]
        == "SIRCANA and our media solutions, our measurement solutions, which are based on deterministic data, can help you do just that."
    )
    assert [i for i, _ in iter_text_list] == list(range(0, 12))
    iter_text_list_chunk = [
        (i, text) for i, text in soundbites_3909774043.chunks[1].iter_text()
    ]
    assert len(iter_text_list_chunk) == 2
    assert iter_text_list_chunk[0] == (
        2,
        "This solution will enable a much broader base of our mutual clients to be able to quantify the impact of their campaigns on offline product sales.",
    )
    assert iter_text_list_chunk[1] == (
        3,
        "And so, as a partnership, we always think about it as better together. And better together, we are able to make our clients better, right?",
    )


async def test_soundbites_keep_only_in_transcript(soundbites_3909774043):
    soundbites = soundbites_3909774043
    transcript = soundbites.transcript.copy()
    chunk = soundbites.chunks[0]
    chunk_soundbite_start_indexes = set(
        [s.start_segment_index for s in chunk.soundbites]
    )
    segment_indexes = set(
        [i for i in transcript.kept_segments if i in chunk_soundbite_start_indexes]
    )
    transcript.kept_segments = segment_indexes
    soundbites_small = soundbites.keep_only_in_transcript(transcript)
    assert len(soundbites_small.soundbites) == len(segment_indexes)

    soundbites_from_chunk = soundbites.keep_only_in_transcript(transcript.chunks[0])
    assert len(soundbites_from_chunk.soundbites) == len(segment_indexes)

    soundbites_from_diff_chunk = soundbites.keep_only_in_transcript(
        transcript.chunks[1]
    )
    assert len(soundbites_from_diff_chunk.soundbites) == 0

    soundbites_chunk_kept = chunk.keep_only_in_transcript(transcript)
    assert len(soundbites_chunk_kept.soundbites) == len(segment_indexes)

    soundbites_chunk_from_diff_chunk = chunk.keep_only_in_transcript(
        transcript.chunks[1]
    )
    assert len(soundbites_chunk_from_diff_chunk.soundbites) == 0

    transcript.kept_segments = set(list(segment_indexes)[:1])
    soundbites_chunk_partial = chunk.keep_only_in_transcript(transcript.chunks[0])
    assert len(soundbites_chunk_partial.soundbites) == 1


async def test_restrict_offset_to_chunk(transcript_15557970):
    transcript_15557970.split_in_chunks(500)
    chunk_1 = transcript_15557970.chunks[1]
    chunk_1_start_seg = chunk_1.chunk_segment_indexes[0]
    chunk_1_end_seg = chunk_1.chunk_segment_indexes[-1]

    offset_in_bounds = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=0,
    )
    offset_partial_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=0,
    )
    offset_partial_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )
    offset_partial_out_of_bounds_both = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 2,
        word_i_end=1,
    )
    expected_offset = offset_in_bounds.model_copy()

    for test_offset in [
        offset_in_bounds,
        offset_partial_out_of_bounds_left,
        offset_partial_out_of_bounds_right,
        offset_partial_out_of_bounds_both,
    ]:
        new_offset = transcript_15557970.restrict_offset_to_chunk(test_offset, chunk_1)
        assert new_offset == expected_offset

    offset_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_start_seg - 1,
        word_i_end=1,
    )
    offset_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_end_seg + 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )

    for test_offset in [offset_out_of_bounds_left, offset_out_of_bounds_right]:
        new_offset = transcript_15557970.restrict_offset_to_chunk(test_offset, chunk_1)
        assert new_offset is None


async def test_keep_only_cuts(transcript_15557970):
    transcript_15557970.split_in_chunks(500)
    chunk_1 = transcript_15557970.chunks[1]
    chunk_1_start_seg = chunk_1.chunk_segment_indexes[0]
    chunk_1_end_seg = chunk_1.chunk_segment_indexes[-1]

    offset_in_bounds = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=1,
        seg_i_end=chunk_1_end_seg,
        word_i_end=len(chunk_1.segments[-1].words) - 1,
    )
    offset_partial_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=0,
    )
    offset_partial_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )
    offset_partial_out_of_bounds_both = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 2,
        word_i_end=1,
    )
    offset_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_start_seg - 1,
        word_i_end=1,
    )
    offset_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_end_seg + 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )

    expected_kept_segments = list(range(chunk_1_start_seg, chunk_1_end_seg + 1))
    expected_words = [
        w
        for s_idx in expected_kept_segments
        for w in transcript_15557970.segments[s_idx].words
    ][1:-1]
    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_in_bounds], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words

    expected_words = [
        w
        for s_idx in expected_kept_segments
        for w in transcript_15557970.segments[s_idx].words
    ]

    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_partial_out_of_bounds_left], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words
    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_partial_out_of_bounds_right], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words
    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_partial_out_of_bounds_both], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words
    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_partial_out_of_bounds_both], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words
    transcript = transcript_15557970.copy()

    expected_words = []
    transcript.keep_only_cuts([offset_out_of_bounds_left], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words
    transcript = transcript_15557970.copy()
    transcript.keep_only_cuts([offset_out_of_bounds_right], from_chunk=chunk_1)
    assert transcript.chunks[1].kept_words == expected_words


async def test_keep_only_cuts(transcript_15557970):
    transcript_15557970.split_in_chunks(500)
    chunk_1 = transcript_15557970.chunks[1]
    chunk_1_start_seg = chunk_1.chunk_segment_indexes[0]
    chunk_1_end_seg = chunk_1.chunk_segment_indexes[-1]

    offset_in_bounds = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=1,
        seg_i_end=chunk_1_end_seg,
        word_i_end=len(chunk_1.segments[-1].words) - 1,
    )
    offset_partial_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=0,
    )
    offset_partial_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_start_seg,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )
    offset_partial_out_of_bounds_both = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 2,
        word_i_end=1,
    )
    offset_out_of_bounds_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 1,
        word_i_start=0,
        seg_i_end=chunk_1_start_seg - 1,
        word_i_end=1,
    )
    offset_out_of_bounds_right = OffsetToCut(
        seg_i_start=chunk_1_end_seg + 1,
        word_i_start=0,
        seg_i_end=chunk_1_end_seg + 1,
        word_i_end=1,
    )

    offset_disjoint_left = OffsetToCut(
        seg_i_start=chunk_1_start_seg - 2,
        word_i_start=0,
        seg_i_end=chunk_1_start_seg - 2,
        word_i_end=2,
    )

    new_offsets = linearize_and_dedupe_offsets(
        [
            offset_in_bounds,
            offset_partial_out_of_bounds_left,
            offset_partial_out_of_bounds_right,
            offset_partial_out_of_bounds_both,
            offset_out_of_bounds_left,
            offset_out_of_bounds_right,
            offset_disjoint_left,
        ]
    )
    assert new_offsets == [
        OffsetToCut(
            seg_i_start=chunk_1_start_seg - 2,
            word_i_start=0,
            seg_i_end=chunk_1_start_seg - 2,
            word_i_end=2,
        ),
        OffsetToCut(
            seg_i_start=chunk_1_start_seg - 1,
            word_i_start=0,
            seg_i_end=chunk_1_end_seg + 2,
            word_i_end=1,
        ),
    ]
