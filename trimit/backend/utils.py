import trimit.utils.conf
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel
from griptape.utils import Stream
import random
import os
from pathlib import Path
from typing import Union
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from trimit.backend.models import Transcript, TranscriptChunk
from trimit.backend.image import image
from trimit.backend.memory import load_memory
from trimit.utils.prompt_engineering import load_prompt_template_as_string
from trimit.utils.rate_limit import rate_limited
from trimit.app import app, AGENT_OUTPUT_CACHE_DIR, LOCAL_AGENT_OUTPUT_CACHE_DIR
from modal import is_local

from griptape.structures import Agent
from griptape.utils import PromptStack
from griptape.rules import Rule
from griptape.tasks import PromptTask
from griptape.config import StructureConfig, StructureGlobalDriversConfig
from griptape.drivers import OpenAiChatPromptDriver
from rapidfuzz.distance.JaroWinkler import normalized_distance
from schema import Schema
import json
import re
import diskcache as dc


if is_local():
    agent_output_cache_dir = LOCAL_AGENT_OUTPUT_CACHE_DIR
else:
    agent_output_cache_dir = AGENT_OUTPUT_CACHE_DIR

global AGENT_OUTPUT_CACHE
AGENT_OUTPUT_CACHE = dc.Cache(agent_output_cache_dir)


class Message(BaseModel):
    role: str
    message: str


ANSI_CODES = {
    "reset": "0",
    "bold": "1",
    "underline": "4",
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}
TEXT_TAG_FORMAT_MAP = {
    "transcript": {"tag": ("yellow", "bold"), "content": ("blue",)},
    "keep": {"tag": ("cyan", "bold"), "content": ("green", "bold")},
}
ANSI_CODE_PREFIX = "\033["


def agent_output_cache_key_from_args(
    json_mode: bool,
    schema: dict | None,
    rules: list[Rule] | None,
    model: str,
    context: str,
    prompt: str | None = None,
    return_formatted: bool = False,
    conversation: list[Message] | None = None,
):
    assert (
        prompt is not None or conversation is not None
    ), "Must provide prompt or conversation"
    prompt_key = "prompt"
    if conversation is not None:
        assert prompt is None, "Cannot provide both prompt and conversation"
        prompt = "\n".join([f"{m.role}: {m.message}" for m in conversation])
        prompt_key = "conversation"
    return (
        f"model-{model}",
        f"json-{json_mode}",
        f"rules-{rules}",
        f"schema-{schema}",
        f"{prompt_key}-{prompt}",
        f"formatted-{return_formatted}",
        f"{context}",
    )


def add_complete_format(text, format_strs: list[str]):
    reset_str = ANSI_CODES["reset"]
    return f"{add_prefix_format(text, format_strs)}{ANSI_CODE_PREFIX}{reset_str}m"


def add_prefix_format(text, format_strs: list[str]):
    ansi_codes_str = ";".join([ANSI_CODES[format_str] for format_str in format_strs])
    return f"{ANSI_CODE_PREFIX}{ansi_codes_str}m{text}"


def strip_carats(text):
    return text.split("<")[-1].split(">")[0]


def find_new_format(text):
    start_format, end_format = [], []
    if text.endswith(">"):
        if text.startswith("</"):
            end_format_tag = strip_carats(text).split("/")[-1]
            end_format = TEXT_TAG_FORMAT_MAP.get(end_format_tag, {}).get("tag", [])
        elif text.startswith("<"):
            start_format_tag = strip_carats(text)
            start_format = TEXT_TAG_FORMAT_MAP.get(start_format_tag, {}).get("tag", [])
    return start_format, end_format


def get_content_format_strs(text):
    return TEXT_TAG_FORMAT_MAP.get(strip_carats(text), {}).get("content", [])


def max_tag_length():
    # add 3 for start carat, end carat, and slash
    return 3 + max([len(k) for k in TEXT_TAG_FORMAT_MAP.keys()])


async def format_text_wrapper(text_iter):
    current_raw_text = ""
    current_format_strs = []
    parent_format_strs = []
    async for chunk in text_iter:
        for char in chunk:
            if "<" == char:
                formatted_text = current_raw_text
                if current_format_strs:
                    formatted_text = add_complete_format(
                        current_raw_text, current_format_strs
                    )
                yield current_raw_text, formatted_text
                current_raw_text = ""

            current_raw_text += char
            new_start_format, new_end_format = find_new_format(current_raw_text)
            if new_start_format:
                formatted_text = add_complete_format(current_raw_text, new_start_format)
                yield current_raw_text, formatted_text
                if current_format_strs:
                    parent_format_strs.append(current_format_strs)
                current_format_strs = get_content_format_strs(current_raw_text)
                current_raw_text = ""
            elif new_end_format:
                formatted_text = add_complete_format(current_raw_text, new_end_format)
                yield current_raw_text, formatted_text
                current_format_strs = (
                    parent_format_strs.pop(-1) if len(parent_format_strs) else []
                )
                current_raw_text = ""
            elif len(current_raw_text) >= max_tag_length():
                formatted_text = current_raw_text
                if current_format_strs:
                    formatted_text = add_complete_format(
                        current_raw_text, current_format_strs
                    )
                yield current_raw_text, formatted_text
                current_raw_text = ""
    if current_raw_text:
        formatted_text = current_raw_text
        if current_format_strs:
            formatted_text = add_complete_format(current_raw_text, current_format_strs)
        yield current_raw_text, formatted_text


def get_agent_output_modal_or_local(*args, **kwargs):
    if is_local():
        return get_agent_output_modal.local(*args, **kwargs)
    else:
        # TODO this isn't getting hydrated
        return get_agent_output_modal.remote_gen.aio(*args, **kwargs)


@app.function(
    is_generator=True,
    allow_concurrent_inputs=100,
    _experimental_boost=True,
    _experimental_scheduler=True,
    retries=3,
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=30,
)
async def get_agent_output_modal(
    prompt: str | None = None,
    conversation: list[Message] | None = None,
    json_mode: bool = False,
    schema: dict | None = None,
    rules: list[Rule] | None = None,
    model: str = "gpt-4o",
    from_cache: bool = True,
    stream: bool = True,
    return_formatted: bool = False,
    **context,
):
    async for output, is_last in get_agent_output(
        prompt=prompt,
        conversation=conversation,
        json_mode=json_mode,
        schema=schema,
        rules=rules,
        model=model,
        from_cache=from_cache,
        stream=stream,
        return_formatted=return_formatted,
        **context,
    ):
        yield output, is_last


async def get_agent_output(
    prompt: str | None = None,
    conversation: list[Message] | None = None,
    json_mode: bool = False,
    schema: dict | None = None,
    rules: list[Rule] | None = None,
    model: str = "gpt-4o",
    from_cache: bool = True,
    stream: bool = True,
    return_formatted: bool = False,
    **context,
):
    cache_key = agent_output_cache_key_from_args(
        prompt=prompt,
        conversation=conversation,
        json_mode=json_mode,
        schema=schema,
        rules=rules,
        model=model,
        return_formatted=return_formatted,
        context=str(context),
    )
    if from_cache and cache_key in AGENT_OUTPUT_CACHE:
        last_output, outputs = AGENT_OUTPUT_CACHE[cache_key]
        for _output in outputs:
            yield _output, False
        yield last_output, True
        return

    async_gen = get_agent_output_no_cache(
        prompt=prompt,
        conversation=conversation,
        json_mode=json_mode,
        schema=schema,
        rules=rules,
        model=model,
        stream=stream,
        return_formatted=return_formatted,
        **context,
    )
    if stream:
        outputs = []
        async for output_chunk in async_gen:
            if isinstance(output_chunk, str):
                yield output_chunk, False
            outputs.append(output_chunk)
        output = None
        if len(outputs):
            output = outputs.pop(-1)
    else:
        output, outputs = await async_gen.__anext__()

    AGENT_OUTPUT_CACHE[cache_key] = (output, outputs)
    yield output, True


@rate_limited(0.5)
async def get_agent_output_no_cache(
    prompt: str | None = None,
    conversation: list[Message] | None = None,
    json_mode: bool = False,
    schema: dict | None = None,
    rules: list[Rule] | None = None,
    model: str = "gpt-4o",
    stream: bool = True,
    return_formatted: bool = False,
    **context,
):
    if prompt is not None:
        task = PromptTask(prompt, context=context)
    elif conversation is not None:
        task = CustomConversationPromptTask(conversation, context=context)
    else:
        raise ValueError("Must provide either prompt or conversation")
    if json_mode:
        if not schema:
            raise ValueError("Must provide schema for JSON output")
        agent = create_structured_agent(schema, rules=rules, model=model, stream=stream)
    else:
        agent = create_agent(summary=False, rules=rules, model=model, stream=stream)
    agent.add_task(task)

    if stream:
        agent = Stream(agent)

        async def run_fn():
            for artifact in agent.run():
                yield artifact.value

    else:

        async def run_fn():
            agent.run()
            yield agent.task.output.value

    outputs = []
    formatted_outputs = []
    async for raw, formatted in format_text_wrapper(run_fn()):
        if stream:
            if return_formatted:
                yield formatted
            else:
                yield raw
        outputs.append(raw)
        formatted_outputs.append(formatted)
    output = "".join(outputs)
    if json_mode:
        output = json.loads(output)
    if stream:
        yield output
    else:
        yield output, formatted_outputs


def get_parsed_transcript_diarization(video):
    parsed = []
    for segment in video.transcription["segments"]:
        speaker = segment.get("speaker", "speaker_UNKNOWN")
        if not speaker.lower().startswith("speaker_"):
            speaker = f"speaker_{speaker}"
        parsed_segment = f'<{speaker.lower()}>{segment["text"]}</{speaker.lower()}>'
        parsed.append(parsed_segment)
    return "".join(parsed)


async def find_leftover_transcript_offsets_using_llm(
    segment_words_without_keep,
    leftover_transcript,
    model="gpt-4o",
    use_agent_output_cache=True,
):
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "offsets": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": [{"type": "integer", "minimum": 0}, {"type": "string"}],
                },
            }
        },
        "required": ["offsets"],
        "additionalProperties": False,
    }
    prompt = load_prompt_template_as_string(
        "find_leftover_transcript_offsets_using_llm"
    )
    sentence_str = " ".join(segment_words_without_keep)
    transcript_formatted_list = [
        f'[{i}, "{w}"]' for i, w in enumerate(leftover_transcript)
    ]
    transcript_formatted_string = ", ".join(transcript_formatted_list)
    output = None
    async for output, is_last in get_agent_output(
        prompt,
        json_mode=True,
        schema=schema,
        model=model,
        sentence=sentence_str,
        transcript=transcript_formatted_string,
        with_cache=use_agent_output_cache,
    ):
        if not is_last:
            yield output, is_last
    assert output is not None
    yield condense_integers([i for i, _ in output["offsets"]]), True


def condense_integers(numbers):
    if not numbers:
        return []

    result = []
    start = numbers[0]

    for i in range(1, len(numbers)):
        # Check if current number is not contiguous with the previous
        if numbers[i] != numbers[i - 1] + 1:
            # Append the previous range
            result.append((start, numbers[i - 1] + 1))
            # Update the start to the current number
            start = numbers[i]

    # Append the last range
    result.append((start, numbers[-1] + 1))

    return result


def condense_single_segment_offsets(
    offsets: list[Union[list[int], tuple[int, int]]]
) -> list[list[int]]:
    new_offsets = []
    current = [offsets[0][0], offsets[0][1]]
    for start, end in offsets[1:]:
        if start == current[1]:
            current[1] = end
        else:
            new_offsets.append(current)
            current = [start, end]
    new_offsets.append(current)
    return new_offsets


def condense_offsets(
    offsets: list["OffsetToCut"], segment_lengths: list[int]
) -> list["OffsetToCut"]:
    # TODO test GroundedOffsetToCut addition and comparators and use them here to simplify
    if not offsets:
        return []
    offsets = sorted(offsets, key=lambda o: (o.seg_i_start, o.word_i_start))
    condensed = []
    current = offsets[0]
    for offset in offsets[1:]:
        if offset.seg_i_start < current.seg_i_end:
            if offset.seg_i_end < current.seg_i_end:
                continue
            elif offset.seg_i_end == current.seg_i_end:
                current.word_i_end = max(current.word_i_end, offset.word_i_end)
            else:
                current.seg_i_end = offset.seg_i_end
                current.word_i_end = offset.word_i_end
        elif offset.seg_i_start == current.seg_i_end:
            if offset.word_i_start <= current.word_i_end:
                if offset.seg_i_end == current.seg_i_end:
                    current.word_i_end = max(current.word_i_end, offset.word_i_end)
                else:
                    current.word_i_end = offset.word_i_end
                    current.seg_i_end = offset.seg_i_end
            else:
                condensed.append(current)
                current = offset
        elif offset.seg_i_start == current.seg_i_end + 1:
            # TODO
            if current.word_i_end == segment_lengths[current.seg_i_end]:
                if offset.word_i_start == 0:
                    current.seg_i_end = offset.seg_i_end
                    current.word_i_end = offset.word_i_end
                else:
                    condensed.append(current)
                    current = offset
            else:
                condensed.append(current)
                current = offset
        else:
            condensed.append(current)
            current = offset
    if (
        len(condensed) == 0
        or current.seg_i_end > condensed[-1].seg_i_end
        or (
            current.seg_i_end == condensed[-1].seg_i_end
            and current.word_i_end > condensed[-1].word_i_end
        )
    ):
        condensed.append(current)
    return condensed


def find_match_stats(s1, s2):
    total_length = max(len(s1), len(s2))
    shorter = min(len(s1), len(s2))
    return sum(c1 == c2 for c1, c2 in zip(s1[:shorter], s2[:shorter])), total_length


def find_leftover_transcript_offsets_fast_for_segment(
    segment: list[str],
    transcript: list[str],
    transcript_start_offset: int,
    match_distance_threshold: float = 0.2,
    max_additional_segment_words_multiplier=0.5,
    # governs how much to prioritize matching near the start of transcript[transcript_start_offset:]
    # vs match distance metric
    start_offset_weight=0.2,
):
    distance_weight = 1 - start_offset_weight
    segment_str = "".join(segment)
    window_sz = len(segment)
    if transcript_start_offset >= len(transcript):
        return -1, transcript_start_offset
    rng = list(range(transcript_start_offset, 1 + len(transcript) - window_sz))
    if len(rng) == 0:
        rng = [transcript_start_offset]
    possible_matches = []
    found_exact = False
    for relative_start_offset in rng:
        relevant_transcript = transcript[
            relative_start_offset : relative_start_offset + window_sz
        ]
        distance = normalized_distance(segment_str, "".join(relevant_transcript))
        if distance < match_distance_threshold:
            possible_matches.append((relative_start_offset, distance))
            if distance == 0:
                found_exact = True
                break

    if len(possible_matches) == 0:
        return -1, transcript_start_offset
    if found_exact:
        return relative_start_offset, relative_start_offset + window_sz

    def sort_key(elt):
        start_offset, distance = elt
        # include start_offset as a tuple to break ties even if start_offset_weight is 0
        start_offset_normalized = start_offset / len(transcript)
        return (
            start_offset_weight * start_offset_normalized + distance_weight * distance,
            start_offset,
        )

    best_start_offset, best_start_distance = min(possible_matches, key=sort_key)
    # TODO potentially multiple offsets?
    end_offset = find_best_end_offset(
        segment=segment,
        transcript=transcript,
        start_offset=best_start_offset,
        seg_len_distance=best_start_distance,
        max_additional_words=int(
            len(segment) * max_additional_segment_words_multiplier
        ),
    )
    return best_start_offset, end_offset


def find_best_end_offset(
    segment, transcript, start_offset, seg_len_distance, max_additional_words
):
    relevant_transcript = transcript[start_offset:]
    best_end_offset = start_offset + len(segment)
    best_distance = seg_len_distance
    start_compare = False
    for i in range(min(len(relevant_transcript), len(segment) + max_additional_words)):
        if start_compare:
            distance = normalized_distance(
                "".join(segment), "".join(relevant_transcript[: i + 1])
            )
            if distance < best_distance:
                best_distance = distance
                best_end_offset = start_offset + i + 1
        if i < len(segment) and segment[i] != relevant_transcript[i]:
            start_compare = True
    return best_end_offset


def find_leftover_transcript_offsets_fast(
    all_segments_to_match,
    transcript,
    match_distance_threshold=0.9,
    max_additional_segment_words_multiplier=0.5,
    start_offset_weight=0.1,
):
    # do each segment in order (need to get segments from higher level function)
    # try to first search for an approximate match, starting from beginning of transcript using window of size length of segment
    # until some threshold is reached.
    # serially, so that this part of the matching segment can be removed from the transcript.
    # then go to the next segment
    offsets = []
    segment_end_offset = 0
    for segment in all_segments_to_match:
        segment_start_offset, segment_end_offset = (
            find_leftover_transcript_offsets_fast_for_segment(
                segment=segment,
                transcript=transcript,
                transcript_start_offset=segment_end_offset,
                match_distance_threshold=match_distance_threshold,
                max_additional_segment_words_multiplier=max_additional_segment_words_multiplier,
                start_offset_weight=start_offset_weight,
            )
        )
        if segment_start_offset > -1 and segment_end_offset > segment_start_offset:
            offsets.append((segment_start_offset, segment_end_offset))
            continue

        # start looking from beginning again, this time with no start_offset_weight
        segment_end_offset = 0
        segment_start_offset, segment_end_offset = (
            find_leftover_transcript_offsets_fast_for_segment(
                segment=segment,
                transcript=transcript,
                transcript_start_offset=segment_end_offset,
                match_distance_threshold=match_distance_threshold,
                max_additional_segment_words_multiplier=max_additional_segment_words_multiplier,
                start_offset_weight=0,
            )
        )
        if segment_start_offset > -1 and segment_end_offset > segment_start_offset:
            offsets.append((segment_start_offset, segment_end_offset))
            continue
        else:
            breakpoint()
            print(f"Failed to find match for segment \"{' '.join(segment)}\"")
    return offsets


def find_leftover_transcript_offsets_fuzzy(
    segment_words_without_keep, leftover_transcript, end_offset=0, memo=None
):
    def add_offset(offsets, offset):
        return [(s + offset, e + offset) for s, e in offsets]

    if len(segment_words_without_keep) == 0 or len(leftover_transcript) == 0:
        return [], 0, 0
    if memo is None:
        memo = {}
    leftover_transcript_str = "".join(leftover_transcript)
    segment_words_without_keep_str = "".join(segment_words_without_keep)

    if (segment_words_without_keep_str, leftover_transcript_str) in memo:
        offsets, matched_chars, n_chars = memo[
            (segment_words_without_keep_str, leftover_transcript_str)
        ]
        return add_offset(offsets, end_offset), matched_chars, n_chars

    # TODO can this be parallelized?
    # TODO can we assume match_pct will be above a threshold and use that to weed out search?
    offsets_to_sort = {}

    empty_result = [], 0, 0
    for segment_end_break in range(len(segment_words_without_keep)):
        start_seg_len = len(segment_words_without_keep) - segment_end_break
        start_segment = segment_words_without_keep[:start_seg_len]

        for start_offset in range(len(leftover_transcript)):
            new_end_offset = min(start_offset + start_seg_len, len(leftover_transcript))

            matched_chars, n_chars = find_match_stats(
                "".join(start_segment),
                "".join(leftover_transcript[start_offset:new_end_offset]),
            )

            seg_offsets = [(start_offset, new_end_offset)]

            end_segment = segment_words_without_keep[start_seg_len:]
            if end_segment:
                best_end_offsets, matched_chars_end, n_chars_end = (
                    find_leftover_transcript_offsets_fuzzy(
                        end_segment,
                        leftover_transcript[new_end_offset:],
                        end_offset=new_end_offset,
                        memo=memo,
                    )
                )
                seg_offsets.extend(best_end_offsets)
                matched_chars += matched_chars_end
                n_chars += n_chars_end

            seg_offsets = condense_single_segment_offsets(seg_offsets)
            pct_matched = matched_chars / n_chars
            if pct_matched == 1.0:
                memo[(segment_words_without_keep_str, leftover_transcript_str)] = (
                    seg_offsets,
                    matched_chars,
                    n_chars,
                )
                return add_offset(seg_offsets, end_offset), matched_chars, n_chars
            if matched_chars > 0:
                offsets_to_sort[(pct_matched, -start_offset, segment_end_break)] = (
                    seg_offsets,
                    matched_chars,
                    n_chars,
                )
    if len(offsets_to_sort) == 0:
        memo[(segment_words_without_keep_str, leftover_transcript_str)] = empty_result
        return empty_result
    best_match_key = max(offsets_to_sort)
    best_match = offsets_to_sort[best_match_key]
    memo[(segment_words_without_keep_str, leftover_transcript_str)] = best_match
    return add_offset(best_match[0], end_offset), best_match[1], best_match[2]


def find_leftover_transcript_offsets(segment_words_without_keep, leftover_transcript):
    for segment_end_break in range(len(segment_words_without_keep)):
        start_seg_len = len(segment_words_without_keep) - segment_end_break
        start_segment = segment_words_without_keep[:start_seg_len]
        start_offset = 0
        found = False
        offsets = []
        end_offset = start_seg_len
        for start_offset in range(len(leftover_transcript)):
            end_offset = start_offset + start_seg_len
            found = start_segment == leftover_transcript[start_offset:end_offset]
            if found:
                break
        if found:
            offsets.append((start_offset, end_offset))
            leftover_offset = start_offset + len(start_segment)
            leftover_transcript = leftover_transcript[leftover_offset:]

            end_segment = segment_words_without_keep[start_seg_len:]
            if not end_segment:
                return offsets
            end_offsets = find_leftover_transcript_offsets(
                end_segment, leftover_transcript
            )
            end_offsets = [
                (s + leftover_offset, e + leftover_offset) for s, e in end_offsets
            ]
            offsets.extend(end_offsets)
            return offsets
    return []


def remove_keep_tags(segment_words):
    try:
        start_keep_index = segment_words.index("<keep>")
    except ValueError:
        start_keep_index = -1

    try:
        end_keep_index = segment_words.index("</keep>")
    except ValueError:
        end_keep_index = len(segment_words)
    return segment_words[start_keep_index + 1 : end_keep_index]


async def get_offsets_to_cut(
    segment_words,
    leftover_transcript_chunk: Transcript | TranscriptChunk,
    using_llm=True,
    model="gpt-4o",
):

    # longest contigous matching subsequences
    # assume segment words can be broken up into pieces that each have an exact match in leftover_transcript, in same order, with just some words in between
    # start with all of segment_words, see if we can find a match
    # then if not, remove the last word, try again starting from beginning going to end
    #  - in this iteration we would wait to find the first part. If we do, we recurse using the rest of leftover_transcript after first part, with leftover segment_words from last word
    #  - otherwise, subtract one from last_offset to use segment_words[:-2], with a recruse if we find it on segment_words[-2:] and leftover_transcript after the index

    segment_words_without_keep = remove_keep_tags(segment_words)
    if using_llm:
        output = None
        async for output, is_last in find_leftover_transcript_offsets_using_llm(
            segment_words_without_keep,
            leftover_transcript_chunk.kept_words,
            model=model,
        ):
            if not is_last:
                yield output, is_last
        assert output is not None
        offsets = output
    else:
        offsets = find_leftover_transcript_offsets(
            segment_words_without_keep, leftover_transcript_chunk.kept_words
        )
    yield leftover_transcript_chunk.full_offsets_from_kept_offsets(offsets), True


def split_on_keep_tags(segment):
    words = []
    found_keep = False
    found_end_keep = False
    for word in segment.split():
        if word.startswith("<keep>"):
            found_keep = True
            words.append(word[:6])
            if len(word) > 6:
                words.append(word[6:])
            continue
        elif word.endswith("</keep>"):
            found_end_keep = True
            if len(word) > 7:
                words.append(word[:-7])
            words.append(word[-7:])
            continue
        elif word == "<keep>":
            found_keep = True
        elif word == "</keep>":
            found_end_keep = True
        words.append(word)
    found_keep = found_keep and found_end_keep
    return words if found_keep else []


async def match_output_to_actual_transcript(
    actual_transcript_chunk: TranscriptChunk | Transcript,
    output: str,
    using_llm=True,
    model="gpt-4o",
) -> TranscriptChunk | Transcript:
    output_transcript = output.split("<transcript>")[1].split("</transcript>")[0]
    leftover_transcript_chunk = actual_transcript_chunk.copy()
    offset_tasks = []
    for segment in re.split("(?<=</keep>)|(?=<keep>)", output_transcript):
        segment_words = split_on_keep_tags(segment)
        if not segment_words:
            continue

        async def offset_task(segment_words, transcript):
            output = None
            async for output, _ in get_offsets_to_cut(
                segment_words, transcript, using_llm=using_llm, model=model
            ):
                continue
            assert output is not None
            return output

        offset_tasks.append(
            offset_task(segment_words, leftover_transcript_chunk.copy())
        )
    task_results = await tqdm_async.gather(*offset_tasks)
    segment_lengths = [len(s.words) for s in leftover_transcript_chunk.segments]
    offsets = condense_offsets(
        [o for segment_offsets in task_results for o in segment_offsets],
        segment_lengths,
    )
    leftover_transcript_chunk.keep_only_cuts(offsets)
    return leftover_transcript_chunk


def match_output_to_actual_transcript_fuzzy(
    actual_transcript_chunk: TranscriptChunk | Transcript, output: str, use_rust=False
) -> TranscriptChunk | Transcript:
    output_transcript = output.split("<transcript>")[1].split("</transcript>")[0]
    leftover_transcript_chunk = actual_transcript_chunk.copy()

    all_words_to_match = []
    for segment in re.split("(?<=</keep>)|(?=<keep>)", output_transcript):
        segment_words = split_on_keep_tags(segment)
        if not segment_words:
            continue
        segment_words_without_keep = remove_keep_tags(segment_words)
        all_words_to_match.extend(segment_words_without_keep)

    fn = find_leftover_transcript_offsets_fuzzy
    if use_rust:
        raise NotImplementedError
        # fn = rust_find_leftover_transcript_offsets_fuzzy_parallel
    kept_offsets, _, _ = fn(all_words_to_match, leftover_transcript_chunk.kept_words)
    kept_offsets, _, _ = fn(all_words_to_match, leftover_transcript_chunk.kept_words)
    offsets = leftover_transcript_chunk.full_offsets_from_kept_offsets(kept_offsets)

    segment_lengths = [len(s.words) for s in leftover_transcript_chunk.segments]
    offsets = condense_offsets(offsets, segment_lengths)
    leftover_transcript_chunk.keep_only_cuts(offsets)
    return leftover_transcript_chunk


# TODO another one of these that allows reordering. Prompt LLM to output the order of segments if it reorders any
# TODO maybe use embeddings to find closest nearest neighbor if no match found
def match_output_to_actual_transcript_fast(
    actual_transcript_chunk: TranscriptChunk | Transcript,
    output: str,
    match_distance_threshold=0.9,
    max_additional_segment_words_multiplier=0.5,
    start_offset_weight=0.1,
    return_offsets=False,
) -> TranscriptChunk | Transcript:
    if "<transcript>" not in output or "</transcript>" not in output:
        if return_offsets:
            return actual_transcript_chunk.copy(), []
        return actual_transcript_chunk.copy()
    output_transcript = output.split("<transcript>")[1].split("</transcript>")[0]
    leftover_transcript_chunk = actual_transcript_chunk.copy()

    all_segments_to_match = []
    for segment in re.split("(?<=</keep>)|(?=<keep>)", output_transcript):
        segment_words = split_on_keep_tags(segment)
        if not segment_words:
            continue
        segment_words_without_keep = remove_keep_tags(segment_words)
        all_segments_to_match.append(segment_words_without_keep)

    # kept_offsets = find_leftover_transcript_offsets_fast(
    text_offsets = find_leftover_transcript_offsets_fast(
        all_segments_to_match,
        # leftover_transcript_chunk.kept_words,
        leftover_transcript_chunk.words,
        match_distance_threshold=match_distance_threshold,
        max_additional_segment_words_multiplier=max_additional_segment_words_multiplier,
        start_offset_weight=start_offset_weight,
    )
    # offsets = leftover_transcript_chunk.full_offsets_from_kept_offsets(kept_offsets)
    offsets = leftover_transcript_chunk.seg_offsets_from_text_offsets(text_offsets)
    # TODO last offset here is beyond transcript chunk, and also not matching
    # even though the very next segment (or 2 after) the second to last one, is the correct match
    # I just changed to use the full trhanscript words instead of kept, since sometimes the LLM returns words not in kept
    # let's see if this plays out
    # TODO why does it not take into account my feedback and always end with "can help you do just that?"
    # Biggest thing is going to be taing user feedback, providing to a new specific prompt, and getting LLM to figure out which specific parts to correct, while keeping the rest constant

    # TODO add back in after fixing
    # segment_lengths = [len(s.words) for s in leftover_transcript_chunk.segments]
    # offsets = condense_offsets(offsets, segment_lengths)
    # TODO think if this erase makes sense here
    leftover_transcript_chunk.erase_cuts()
    leftover_transcript_chunk.keep_only_cuts(offsets)
    if return_offsets:
        return leftover_transcript_chunk, offsets
    return leftover_transcript_chunk


def remove_boundary_tags(tag, text):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    text = text.strip()
    if text.startswith(start_tag):
        text = text[len(start_tag) :]
    if text.endswith(end_tag):
        text = text[: -len(end_tag)]
    return text


async def print_with_chunk_delay(text, delay=None):
    if delay is None:
        from trimit.backend.conf import CONF

        delay = CONF["chunk_delay"]
    for chunk in text:
        print(chunk, end="", flush=True)
        await asyncio.sleep(delay)


async def get_user_feedback(
    prompt,
    with_provided_user_feedback: list[str] | None = None,
    ask_for_feedback: bool = True,
):
    await print_with_chunk_delay(prompt)
    if with_provided_user_feedback and len(with_provided_user_feedback):
        user_feedback = with_provided_user_feedback.pop(0)
        await print_with_chunk_delay("\n" + user_feedback + "\n")
    elif ask_for_feedback:
        user_feedback = input("\n")
    else:
        await print_with_chunk_delay("\nUser feedback skipped for current run")
        user_feedback = ""
    return user_feedback


async def get_user_feedback_yielding(prompt):
    user_feedback = yield prompt
    yield user_feedback
    return


def save_transcript_to_disk(
    output_dir: str,
    transcript: Transcript | TranscriptChunk,
    stage_num: int = 0,
    timeline_name: str = "",
    save_text_file=True,
):
    if timeline_name:
        output_dir = os.path.join(output_dir, timeline_name)
    stage_num_suffix = f"_{stage_num}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    transcript_cache_plaintext_file = os.path.join(
        output_dir, f"transcript{stage_num_suffix}.txt"
    )
    transcript_cache_pickle_file = os.path.join(
        output_dir, f"transcript{stage_num_suffix}.p"
    )
    transcript.save(transcript_cache_pickle_file)
    if save_text_file:
        with open(transcript_cache_plaintext_file, "w") as f:
            f.write(transcript.text)
        return transcript_cache_pickle_file, transcript_cache_plaintext_file
    return transcript_cache_pickle_file, None


def load_latest_transcript_from_disk(
    transcript_cache_file_dir: str, stage_num: int | str = "latest"
):
    if stage_num == "latest":
        saved_stages = [
            f.split("transcript_")[1].split(".")[0]
            for f in os.listdir(transcript_cache_file_dir)
            if f.startswith("transcript_")
        ]
        latest_stage = -1
        for stage in saved_stages:
            try:
                stage_parsed = int(stage)
            except:
                continue
            else:
                if stage_parsed > latest_stage:
                    latest_stage = stage_parsed
        if latest_stage == -1:
            raise ValueError("No saved transcripts found")
        stage_num_suffix = f"_{stage_parsed}"
    else:
        stage_num_suffix = f"_{stage_num}"
    transcript_cache_pickle_file = os.path.join(
        transcript_cache_file_dir, f"transcript{stage_num_suffix}.p"
    )
    return Transcript.load_from_file(transcript_cache_pickle_file)


def get_soundbite_rule(soundbites: Union["Soundbites", "SoundbitesChunk"]):
    formatted = "\n".join(
        f"<soundbite>{text}</soundbite>" for _, text in soundbites.iter_text()
    )
    return [
        Rule(
            "If at all possible, try to include as many of the following hand-picked soundbites taken from the transcript in your <keep></keep> segments. "
            f"Note that these might not be currently present in the transcript, but they are always viable options to include: \n{formatted}"
        )
    ]


def remove_off_screen_speakers(on_screen_speakers: list[str], transcript: Transcript):
    on_screen_speakers = [s.lower() for s in on_screen_speakers]
    on_screen_transcript = transcript.copy()
    for i, segment in enumerate(transcript.segments):
        if segment.speaker.lower() not in on_screen_speakers:
            on_screen_transcript.remove_segment(i)
    return on_screen_transcript


def segment_partials_to_final_transcript(segment_partials):
    return "\n".join([" ".join(s) for t in segment_partials for s in t])


def desired_words_from_length(length_seconds, wpm=130):
    return int(round(length_seconds / 60 * wpm))


async def remove_soundbites(soundbites, max_soundbites):
    soundbites = soundbites.copy()
    prompt = load_prompt_template_as_string("remove_soundbites")
    schema = Schema({"soundbite_indexes_to_keep": [int]}).json_schema(
        "RemoveSoundbites"
    )
    output = None
    async for output, is_last in get_agent_output(
        prompt,
        schema=schema,
        json_mode=True,
        soundbites=[(i, text) for i, text in soundbites.iter_text()],
        max_soundbites=max_soundbites,
    ):
        if not is_last:
            yield output, is_last
    assert isinstance(output, dict)
    if "soundbite_indexes_to_keep" not in output:
        raise ValueError("Expected 'soundbite_indexes_to_keep' in output")
    to_keep = output["soundbite_indexes_to_keep"]
    if len(to_keep) > max_soundbites:
        soundbites.soundbites = random.sample(soundbites.soundbites, max_soundbites)
    else:
        soundbites.soundbites = [soundbites.soundbites[i] for i in to_keep]
    yield soundbites, True


def parse_partials_to_redo_from_agent_output(agent_output: dict, n):
    partials_to_redo = agent_output.get("chunks_to_redo", [])
    if not partials_to_redo:
        return []
    if len(partials_to_redo) != n:
        print(f"Expected {n} partials to redo, but got {len(partials_to_redo)}")
        return [True] * n
    return partials_to_redo


def parse_relevant_user_feedback_list_from_agent_output(
    agent_output: dict, n: int, user_feedback: str
):
    relevant_user_feedback_list = agent_output.get("relevant_user_feedback_list", [])
    if len(relevant_user_feedback_list) != n:
        print(
            f"Expected {n} user feedback strings, but got {len(relevant_user_feedback_list)}"
        )
        return [user_feedback] * n
    return relevant_user_feedback_list


def remove_retry_suffix(step_name_with_retry):
    return step_name_with_retry.split("_retry_")[0]


def add_retry_suffix(step_name, retry_num):
    return f"{step_name}_retry_{retry_num}"


def create_json_rule(schema: dict):
    return Rule(value=f"Write your output in JSON using the schema: {schema}")


class CustomConversationPromptTask(PromptTask):
    def __init__(self, conversation: list[Message], *args, **kwargs):
        super().__init__(conversation[-1].message, *args, **kwargs)
        self.conversation = conversation

    @property
    def prompt_stack(self) -> PromptStack:
        stack = PromptStack()
        for message in self.conversation:
            stack.add_input(message.message, message.role)
        return stack


def create_agent(
    model="gpt-4o",
    json=False,
    summary=False,
    rules: list[Rule] | None = None,
    stream: bool = False,
):
    if json and summary:
        raise ValueError("Can't have both json and summary")
    driver_kwargs = {"model": model}
    if json:
        driver_kwargs["response_format"] = "json_object"
    if stream:
        driver_kwargs["stream"] = True

    memory = None
    if summary:
        memory = load_memory(summary=summary)
    return Agent(
        config=StructureConfig(
            global_drivers=StructureGlobalDriversConfig(
                prompt_driver=OpenAiChatPromptDriver(**driver_kwargs)
            )
        ),
        conversation_memory=memory,
        rules=rules,
    )


def create_structured_agent(
    schema, model="gpt-4o", rules: list[Rule] | None = None, stream: bool = False
):
    if rules is None:
        rules = []
    else:
        rules = rules[:]
    rules.append(create_json_rule(schema))
    agent = create_agent(
        model=model, json=True, summary=False, rules=rules, stream=stream
    )
    return agent


def parse_stage_num_from_step_name(step_name):
    if step_name.startswith("stage_"):
        return int(step_name.split("_")[1])


def stage_key_for_step_name(step_name, stage_num):
    return f"stage_{stage_num}_{step_name}"
