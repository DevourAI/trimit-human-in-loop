from trimit.backend.utils import (
    find_leftover_transcript_offsets_using_llm,
    match_output_to_actual_transcript,
    condense_offsets,
    find_leftover_transcript_offsets_fast,
    match_output_to_actual_transcript_fast,
)
import pytest

pytestmark = pytest.mark.asyncio(scope="session")


async def test_find_leftover_transcript_offsets_using_llm():
    words = (
        "I'm the VP GM at CVS Media Exchange "
        "We are here building a coalition and partnership in collaboration with Pinterest."
    ).split()
    transcript = (
        "I'm the VP GM over at CVS Media Exchange, which is, well, CMX as we like to refer to it. "
        "We're the retail media network for CVS Health.  We are here at ramp-up and we're also announcing "
        "here at ramp-up that we're building a coalition and partnership in collaboration with Pinterest."
    ).split()
    output = find_leftover_transcript_offsets_using_llm(
        words, transcript, model="gpt-4o"
    )
    assert output == [(0, 4), (5, 9), (28, 31), (42, 51)]


async def test_condense_offsets(transcript_15557970):
    # TODO actually test this
    offsets = []
    chunk = transcript_15557970.chunks[0]
    segment_lengths = [len(s.words) for s in chunk.segments]
    offsets = condense_offsets(offsets, segment_lengths)


async def test_find_leftover_transcript_offsets_fast_simple():
    segments = [["examples", "words"], ["without", "keep"]]
    transcript = ["leftover", "words", "transcript", "additional"]
    kept_offsets = find_leftover_transcript_offsets_fast(
        all_segments_to_match=segments,
        transcript=transcript,
        match_distance_threshold=0.5,
        max_additional_segment_words_multiplier=0,
    )
    assert kept_offsets == [(1, 2)]


async def test_find_leftover_transcript_offsets_fast_complex():
    # TODO need to tell LLM to split disjoint segments into their own <keep> tags
    segments = [
        "I'm the VP GM at CVS Media Exchange ".split(),
        "We are here ".split(),
        "building a coalition and partnership in collaboration with Pinterest.".split(),
    ]
    transcript = (
        "I'm the VP GM over at CVS Media Exchange, which is, well, CMX as we like to refer to it. "
        "We're the retail media network for CVS Health.  We are here at ramp-up and we're also announcing "
        "here at ramp-up that we're building a coalition and partnership in collaboration with Pinterest."
    ).split()
    kept_offsets = find_leftover_transcript_offsets_fast(
        all_segments_to_match=segments * 2,
        transcript=transcript * 2,
        match_distance_threshold=0.9,
        max_additional_segment_words_multiplier=0.5,
        start_offset_weight=0.1,
    )
    assert kept_offsets == [(0, 9), (28, 31), (42, 51), (51, 60), (79, 82), (93, 102)]


async def test_match_output_to_actual_transcript(transcript_15557970):
    ai_output = """
<transcript>
<keep>I'm the VP GM at CVS Media Exchange, which is, well, CMX as we like to refer to it.</keep>
<keep>And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.</keep>
So the way in which we operate, CMX operates, we target our extra care consumers or really use that as our baseline for the way in which we build our business.
Our extra care consumers, which are incredibly loyal to our CVS brand, they have a tenure of over 10 years. We have 74 million of them who are engaging with us.
CVS also has around about 9,000 locations. We have close to 5 million consumers who come into and through our stores every single day.
There is an awful amount of different engagement and ways in which we can communicate with the consumer.
<keep>And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.</keep>
That's great. And so when you were thinking about new goals or your vision for CMX as it grew,
What was the role of your partnership with LiveRamp in that? What goals, vision did you have for CMX that led you to partnering with LiveRamp?
<keep>Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX, but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving
and growing our business and for example given our extra care loyalty consumers that I mentioned around
about 74 million of them ensuring that we can utilize those audience and first party data assets in in
clean room environment in a trusted clean room environment. </keep>
<keep>LiveRamp is a really core component to the way in which we're thinking about our business, not just today, but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that partnership.</keep>
Do you want to tell us a little bit more about the role that clean rooms play in CMX and What do they allow you to do that you couldn't do before?
</transcript>
"""
    expected = """
I'm the VP GM at CVS Media Exchange, which is, well, CMX as we like to refer to it.
And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.
And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.
Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX,
but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving and growing our
business and for example given our extra care loyalty consumers that I mentioned around about 74 million
of them ensuring that we can utilize those audience and first party data assets in in clean room environment in a trusted clean room environment.
LiveRamp is a really core component to the way in which we're thinking about our business, not just today,
but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that partnership.
"""
    leftover_transcript_chunk = await match_output_to_actual_transcript(
        transcript_15557970.chunks[0], ai_output
    )
    assert leftover_transcript_chunk.kept_words == expected.strip().split()


async def test_match_output_to_actual_transcript_fast(transcript_15557970):
    ai_output = """
<transcript>
<keep>I'm the VP GM at CVS Media Exchange, which is, well, CMX as we like to refer to it.</keep>
<keep>And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.</keep>
So the way in which we operate, CMX operates, we target our extra care consumers or really use that as our baseline for the way in which we build our business.
Our extra care consumers, which are incredibly loyal to our CVS brand, they have a tenure of over 10 years. We have 74 million of them who are engaging with us.
CVS also has around about 9,000 locations. We have close to 5 million consumers who come into and through our stores every single day.
There is an awful amount of different engagement and ways in which we can communicate with the consumer.
<keep>And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.</keep>
That's great. And so when you were thinking about new goals or your vision for CMX as it grew,
What was the role of your partnership with LiveRamp in that? What goals, vision did you have for CMX that led you to partnering with LiveRamp?
<keep>Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX, but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving
and growing our business and for example given our extra care loyalty consumers that I mentioned around
about 74 million of them ensuring that we can utilize those audience and first party data assets in in
clean room environment in a trusted clean room environment. </keep>
<keep>LiveRamp is a really core component to the way in which we're thinking about our business, not just today, but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that partnership.</keep>
Do you want to tell us a little bit more about the role that clean rooms play in CMX and What do they allow you to do that you couldn't do before?
</transcript>
"""
    expected = """
I'm the VP GM over at CVS Media Exchange, which is, well, CMX as we like to refer to it.
And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.
And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.
Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX,
but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving and growing our
business and for example given our extra care loyalty consumers that I mentioned around about 74 million
of them ensuring that we can utilize those audience and first party data assets in in clean room environment in a trusted clean room environment.
LiveRamp is a really core component to the way in which we're thinking about our business, not just today,
but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that partnership.
"""
    leftover_transcript_chunk = match_output_to_actual_transcript_fast(
        transcript_15557970.chunks[0],
        ai_output,
        match_distance_threshold=0.9,
        max_additional_segment_words_multiplier=0.5,
        start_offset_weight=0.1,
    )
    assert leftover_transcript_chunk.kept_words == expected.strip().split()


async def test_match_output_to_actual_transcript_fast_degenerate(transcript_15557970):
    ai_output = """
<transcript></transcript>
"""
    expected = """"""
    leftover_transcript_chunk = match_output_to_actual_transcript_fast(
        transcript_15557970.chunks[0],
        ai_output,
        match_distance_threshold=0.9,
        max_additional_segment_words_multiplier=0.5,
        start_offset_weight=0.1,
    )
    assert leftover_transcript_chunk.kept_words == expected.strip().split()


async def test_match_output_to_actual_transcript_fast_agent_cutoff(transcript_15557970):
    ai_output = """
<transcript>
<keep>I'm the VP GM at CVS Media Exchange, which is, well, CMX as we like to refer to it.</keep>
<keep>And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.</keep>
So the way in which we operate, CMX operates, we target our extra care consumers or really use that as our baseline for the way in which we build our business.
Our extra care consumers, which are incredibly loyal to our CVS brand, they have a tenure of over 10 years. We have 74 million of them who are engaging with us.
CVS also has around about 9,000 locations. We have close to 5 million consumers who come into and through our stores every single day.
There is an awful amount of different engagement and ways in which we can communicate with the consumer.
<keep>And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.</keep>
That's great. And so when you were thinking about new goals or your vision for CMX as it grew,
What was the role of your partnership with LiveRamp in that? What goals, vision did you have for CMX that led you to partnering with LiveRamp?
<keep>Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX, but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving
and growing our business and for example given our extra care loyalty consumers that I mentioned around
about 74 million of them ensuring that we can utilize those audience and first party data assets in in
clean room environment in a trusted clean room environment. </keep>
<keep>LiveRamp is a really core component to the way in which we're thinking about our business, not just today, but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that
"""
    expected = """
I'm the VP GM over at CVS Media Exchange, which is, or CMX as we like to refer to it.
And as we think about servicing our customers and putting customers at the heart of everything that we do,
CMX really sort of brings that experience to life and that connectivity with our consumers.
And CMX is at the heart and center of how we think about consumer journey and engaging consumers,
but also connecting them with brands that they would normally purchase or helping them discover new brands as well.
Yeah, so, and LiveRamp has been a incredible and strong partner of not just CMX,
but CVS Health for quite some time.
We see our partnership with LiveRamp as really core to the way in which we are driving and growing our
business and for example given our extra care loyalty consumers that I mentioned around about 74 million
of them ensuring that we can utilize those audience and first party data assets in in clean room environment in a trusted clean room environment.
LiveRamp is a really core component to the way in which we're thinking about our business, not just today,
but for the future as well.
So we'll continue to lean in from a LiveRamp perspective, but we value that partnership.
Do you want to tell us a little bit more about the role that clean rooms play in CMX and
What do they allow you to do that you couldn't do before?
Yeah, so clean rooms and where clean rooms sit in our overall stack are a really, really important part of our overall delivery of audiences, delivery of insights, delivery of measurement as well, against that loyal customer base that I've talked about, which is extricate.
"""
    leftover_transcript_chunk = match_output_to_actual_transcript_fast(
        transcript_15557970.chunks[0],
        ai_output,
        match_distance_threshold=0.9,
        max_additional_segment_words_multiplier=0.5,
        start_offset_weight=0.1,
    )
    assert leftover_transcript_chunk.kept_words == expected.strip().split()
