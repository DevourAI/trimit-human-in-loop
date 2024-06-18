import json
from IPython.display import Video
from fastapi.encoders import jsonable_encoder
from trimit.models.models import StepKey
from pydantic import BaseModel, Field, model_serializer
from typing import Callable, Optional, Union
import pickle
from trimit.utils.misc import union_list_of_intervals
import copy


class WordInfo(BaseModel):
    start: float
    end: float
    word: str


class Cut(BaseModel):
    words: list[WordInfo]
    segment: "TranscriptSegment"
    is_kept: Optional[bool] = True

    @property
    def start(self):
        return self.words[0].start if len(self.words) else 0

    @property
    def end(self):
        return self.words[-1].end if len(self.words) else -1

    @property
    def text(self):
        return " ".join([wi.word for wi in self.words])


class TranscriptSegment(BaseModel):
    speaker: str
    start: float
    end: float
    word_start_ends: list[WordInfo]
    cut_segments: list[tuple[int, int]]

    @classmethod
    def create(cls, words, speaker, start, end, word_start_ends, cut_segments=None):
        speaker = speaker
        start = start
        end = end
        try:
            word_start_ends = [
                WordInfo(start=start, end=end, word=word)
                for word, (start, end) in zip(words, word_start_ends)
            ]
        except Exception as e:
            raise
            breakpoint()
        # TODO rename cut_segments to kept_segments
        cut_segments = cut_segments if cut_segments is not None else [(0, len(words))]
        return TranscriptSegment(
            speaker=speaker,
            start=start,
            end=end,
            word_start_ends=word_start_ends,
            cut_segments=cut_segments,
        )

    @property
    def words(self):
        return [w.word for w in self.word_start_ends]

    @property
    def keep_indexes(self):
        return set([i for s, e in self.cut_segments for i in range(s, e)])

    @property
    def remove_indexes(self):
        all_indexes = set(range(len(self.words)))
        return all_indexes - self.keep_indexes

    @classmethod
    def from_video_transcription_segment(cls, segment: dict):
        words = [w["word"] for w in segment["words"]]
        speaker = segment["speaker"]
        start = segment["start"]
        end = segment["end"]
        word_start_ends = [(w["start"], w["end"]) for w in segment["words"]]
        return cls.create(
            words=words,
            speaker=speaker,
            start=start,
            end=end,
            word_start_ends=word_start_ends,
        )

    @property
    def kept_words(self):
        kept = []
        for start, end in self.cut_segments:
            kept.extend(self.words[start:end])
        return kept

    def kept_words_with_start_end(self, start_word_index=None, end_word_index=None):
        kept = []
        if start_word_index is None:
            start_word_index = 0
        if end_word_index is None:
            end_word_index = len(self.words)
        for start, end in self.cut_segments:
            if start >= end_word_index:
                continue
            if end <= start_word_index:
                continue

            start_to_use = start_word_index
            if start >= start_word_index:
                start_to_use = start
            end_to_use = end_word_index
            if end <= end_word_index:
                end_to_use = end
            if end_to_use > start_to_use:
                kept.extend(self.words[start_to_use:end_to_use])
        return kept

    def text_with_start_end(self, start_word_index=None, end_word_index=None):
        return " ".join(
            [
                w
                for w in self.kept_words_with_start_end(
                    start_word_index=start_word_index, end_word_index=end_word_index
                )
                if w
            ]
        )

    def word_i_kept(self, word_i):
        for start, end in self.cut_segments:
            if word_i >= start and word_i < end:
                return True
        return False

    def cuts_with_start_end(self, start_word_index=None, end_word_index=None):
        cut_list = []
        cur_segment = Cut(words=[], segment=self)
        is_kept = False
        if start_word_index is None:
            start_word_index = 0
        if end_word_index is None:
            end_word_index = len(self.words)

        for i, word_info in enumerate(self.word_start_ends):
            if i < start_word_index:
                pass
            elif i >= end_word_index:
                if is_kept and len(cur_segment.words):
                    is_kept = False
                    cur_segment.is_kept = True
                    cut_list.append(cur_segment)
                    cur_segment = Cut(words=[], segment=self)
            elif self.word_i_kept(i) and not is_kept:
                if len(cur_segment.words):
                    cur_segment.is_kept = False
                    cut_list.append(cur_segment)
                    cur_segment = Cut(words=[], segment=self)
                is_kept = True
            elif not self.word_i_kept(i) and is_kept:
                if len(cur_segment.words):
                    cur_segment.is_kept = True
                    cut_list.append(cur_segment)
                    cur_segment = Cut(words=[], segment=self)
                is_kept = False
            cur_segment.words.append(word_info)
        if len(cur_segment.words):
            cur_segment.is_kept = is_kept
            cut_list.append(cur_segment)
        return cut_list

    def iter_cuts(self):
        return self.cuts_with_start_end(None, None)

    def kept_cuts_with_start_end(self, start_word_index=None, end_word_index=None):
        return [
            c
            for c in self.cuts_with_start_end(
                start_word_index=start_word_index, end_word_index=end_word_index
            )
            if c.is_kept
        ]

    @property
    def text(self):
        return " ".join([w for w in self.kept_words if w])

    @property
    def words_with_keep_tags(self):
        words = []
        in_kept_segment = False
        for word_i, word in enumerate(self.words):
            if word_i in self.keep_indexes:
                if not in_kept_segment:
                    words.append("<keep>")
                    in_kept_segment = True
                words.append(word)
            else:
                if in_kept_segment:
                    words.append("</keep>")
                    in_kept_segment = False
                words.append(word)
        if in_kept_segment:
            words.append("</keep>")
        return words

    @property
    def words_with_keep_and_remove_tags(self):
        words = []
        in_kept_segment = False
        for word_i, word in enumerate(self.words):
            if word_i in self.keep_indexes:
                if word_i == 0:
                    words.append("<keep>")
                elif not in_kept_segment:
                    words.extend(["</remove>", "<keep>"])
                in_kept_segment = True
            else:
                if word_i == 0:
                    words.append("<remove>")
                elif in_kept_segment:
                    words.extend(["</keep>", "<remove>"])
                in_kept_segment = False
            words.append(word)
        if words[-1] in ["<keep>", "<remove>"]:
            words.pop()
        elif words[-1] not in ["</keep>", "</remove>"]:
            if in_kept_segment:
                words.append("</keep>")
            else:
                words.append("</remove>")
        return words

    def cut(self, start_index, end_index):
        # TODO: perhaps this whole structure would be easier to do in a subtractive as opposed to additive way
        # so this cut would remove parts of the whole (and then we don't have to call erase_cuts() first)
        # And we could have the AI add <cut> tags instead of <keep> tags
        if end_index < start_index:
            raise ValueError("End index must be greater than start index")
        if start_index >= len(self.words) or end_index > len(self.words):
            raise ValueError("Cut index out of range")
        if self.cut_segments == [(0, len(self.words))]:
            self.cut_segments = [(start_index, end_index)]
            return
        # TODO I had to add this in when testing the circana video with the following error. why?
        # File "/Users/ben/storybook/storybook/linear_workflow/cut_transcript.py", line 514, in cut_transcript
        #   new_cut_transcript_chunk = await cut_partial_transcript(
        #                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # File "/Users/ben/storybook/storybook/linear_workflow/cut_transcript.py", line 429, in cut_partial_transcript
        #   return match_output_to_actual_transcript(partial_on_screen_transcript, output)
        #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # File "/Users/ben/storybook/storybook/linear_workflow/cut_transcript.py", line 344, in match_output_to_actual_transcript
        #   leftover_transcript_chunk.keep_only_cuts(offsets)
        # File "/Users/ben/storybook/storybook/linear_workflow/models.py", line 297, in keep_only_cuts
        #   self.transcript.keep_only_cuts(offsets, from_chunk=self)
        # File "/Users/ben/storybook/storybook/linear_workflow/models.py", line 659, in keep_only_cuts
        #   self.segments[seg_i].cut(0, len(self.segments[seg_i].words))
        # File "/Users/ben/storybook/storybook/linear_workflow/models.py", line 94, in cut
        #   raise ValueError("Cut segment overlaps with existing cut segment")
        #   ValueError: Cut segment overlaps with existing cut segment
        elif start_index == 0 and end_index == len(self.words):
            self.cut_segments = [(0, len(self.words))]
            return
        if self.cut_segments:
            if end_index < self.cut_segments[0][0]:
                self.cut_segments.insert(0, (start_index, end_index))
                return
            elif start_index >= self.cut_segments[-1][1]:
                self.cut_segments.append((start_index, end_index))
                return
        i = 0
        for i, (_start, _end) in enumerate(self.cut_segments):
            if end_index <= _start:
                break
            if start_index >= _start and start_index < _end:
                # TODO my theory is that offsets in match_output are overlapping
                # or that somehow the previous cuts aren't getting erased
                raise ValueError(
                    f"Cut segment overlaps with existing cut segment existing_cut_segments={self.cut_segments} start_index={start_index} end_index={end_index}"
                )
            elif end_index > _start and end_index <= _end:
                raise ValueError(
                    f"Cut segment overlaps with existing cut segment existing_cut_segments={self.cut_segments} start_index={start_index} end_index={end_index}"
                )
            elif start_index >= _end:
                if i < len(self.cut_segments) and _end >= self.cut_segments[i + 1][0]:
                    raise ValueError(
                        f"Cut segment overlaps with existing cut segment existing_cut_segments={self.cut_segments} start_index={start_index} end_index={end_index}"
                    )
                break
        self.cut_segments.insert(i, (start_index, end_index))

    def union_with(self, other_segment: "TranscriptSegment"):
        if not other_segment.cut_segments:
            return
        if max(other_segment.cut_segments, key=lambda x: x[1])[1] > len(self.words):
            raise ValueError(
                f"Other cut segment out of range (len(self.words)=={len(self.words)}): {other_segment.cut_segments}"
            )
        assert isinstance(
            self.cut_segments, list
        ), f"self.cut_segments is not a list: {self.cut_segments}"
        assert isinstance(
            other_segment.cut_segments, list
        ), f"other_segment.cut_segments is not a list: {other_segment.cut_segments}"
        self.cut_segments = union_list_of_intervals(
            self.cut_segments, other_segment.cut_segments
        )

    def _create_split_segments(self, split_word_index, cut_segment_index):
        left_segments = self.cut_segments[:cut_segment_index]
        left_start = self.start
        left_end = self.word_start_ends[split_word_index][1]
        left_words = self.words[:split_word_index]
        left_word_start_ends = self.word_start_ends[:split_word_index]
        left_n_words = len(left_words)

        right_segments = []
        if len(self.cut_segments) and cut_segment_index < len(self.cut_segments):
            right_segments = [
                (s - left_n_words, e - left_n_words)
                for s, e in self.cut_segments[cut_segment_index:]
            ]
        right_start = self.word_start_ends[split_word_index][0]
        right_end = self.end
        right_words = self.words[split_word_index:]
        right_word_start_ends = self.word_start_ends[split_word_index:]
        return (
            TranscriptSegment.create(
                left_words,
                self.speaker,
                left_start,
                left_end,
                left_word_start_ends,
                left_segments,
            ),
            TranscriptSegment.create(
                right_words,
                self.speaker,
                right_start,
                right_end,
                right_word_start_ends,
                right_segments,
            ),
        )

    def split(self, index):
        if index >= len(self.words):
            raise ValueError("Index out of range")

        # split falls before first cut segment
        if len(self.cut_segments) == 0 or index < self.cut_segments[0][0]:
            return self._create_split_segments(index, 0)

        for seg_i, (start_i, end_i) in enumerate(self.cut_segments):
            # split falls on an existing cut boundary
            if start_i == index:
                return self._create_split_segments(index, seg_i)
            # split falls in between an existing cut segment
            elif start_i < index and end_i > index:
                self.cut_segments = (
                    self.cut_segments[:seg_i]
                    + [(start_i, index), (index, end_i)]
                    + self.cut_segments[seg_i + 1 :]
                )
                # we add remove seg_i, add two new cut segments and split from the second one onward
                return self._create_split_segments(index, seg_i + 1)
            # split falls in between two cut segments
            elif index >= end_i and (
                seg_i < len(self.cut_segments)
                and index < self.cut_segments[seg_i + 1][0]
            ):
                return self._create_split_segments(index, seg_i + 1)
        # split falls beyond last cut segment
        return self._create_split_segments(index, len(self.cut_segments))

    def copy(self):
        return TranscriptSegment(
            speaker=self.speaker,
            start=self.start,
            end=self.end,
            word_start_ends=self.word_start_ends[:],
            cut_segments=self.cut_segments[:],
        )

    @property
    def state(self):
        return {
            "words": self.words,
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "word_start_ends": self.word_start_ends,
            "cut_segments": self.cut_segments,
        }

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.state, f)

    @classmethod
    def load_from_file(cls, file):
        with open(file, "rb") as f:
            state = pickle.load(f)
        return cls.load_from_state(state)

    @classmethod
    def load_from_state(cls, state):
        wse = state.get("word_start_ends")
        if wse:
            if not isinstance(wse[0], WordInfo):
                try:
                    word_start_ends = [WordInfo(**w) for w in state["word_start_ends"]]
                except Exception as e:
                    return cls.create(**state)
                else:
                    state["word_start_ends"] = word_start_ends
            return cls(
                speaker=state["speaker"],
                start=state["start"],
                end=state["end"],
                word_start_ends=state["word_start_ends"],
                cut_segments=state["cut_segments"],
            )
        else:
            return cls.create(**state)


class TranscriptPartitionBoundary(BaseModel):
    segment_index: int
    word_index: int


class OffsetToCut(BaseModel):
    seg_i_start: int
    word_i_start: int
    seg_i_end: int
    word_i_end: int

    def __eq__(self, other):
        return (
            self.seg_i_start == other.seg_i_start
            and self.seg_i_end == other.seg_i_end
            and self.word_i_start == other.word_i_start
            and self.word_i_end == other.word_i_end
        )

    def __lt__(self, other):
        return self.seg_i_start < other.seg_i_start or (
            self.seg_i_start == other.seg_i_start
            and self.word_i_start < other.word_i_start
        )

    def __gt__(self, other):
        return self.seg_i_end > other.seg_i_end or (
            self.seg_i_start == other.seg_i_start
            and self.word_i_start > other.word_i_start
        )

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __ne__(self, other):
        return not self == other


class GroundedOffsetToCut(OffsetToCut):
    segment_lengths: list[int]

    def __add__(self, i: int):
        left_in_segment = self.segment_lengths[self.seg_i_end] - self.word_i_end
        if i < left_in_segment:
            return self.custom_copy(word_i_end=self.word_i_end + i)
        elif i == left_in_segment:
            if self.seg_i_end + 1 > len(self.segment_lengths):
                raise ValueError("Index out of range")
            return self.custom_copy(seg_i_end=self.seg_i_end + 1, word_i_end=0)
        else:
            new_offset = self.custom_copy()
            i = i - left_in_segment
            for seg_i_offset, segment_length in enumerate(
                self.segment_lengths[self.seg_i_end + 1 :]
            ):
                seg_i = seg_i_offset + self.seg_i_end + 1
                if i < segment_length:
                    new_offset.seg_i_end = seg_i
                    new_offset.word_i_end = i
                    return new_offset
                i -= segment_length

    def custom_copy(
        self, seg_i_start=None, word_i_start=None, seg_i_end=None, word_i_end=None
    ):
        if seg_i_start is None:
            seg_i_start = self.seg_i_start
        if word_i_start is None:
            word_i_start = self.word_i_start
        if seg_i_end is None:
            seg_i_end = self.seg_i_end
        if word_i_end is None:
            word_i_end = self.word_i_end
        return GroundedOffsetToCut(
            seg_i_start=seg_i_start,
            word_i_start=word_i_start,
            seg_i_end=seg_i_end,
            word_i_end=word_i_end,
            segment_lengths=self.segment_lengths,
        )


class TranscriptChunk:
    # allows us to easily get the text of a chunk of the transcript
    # while maintaining references to original segment indexes and kept/cut words
    def __init__(self, chunk_segment_indexes, chunk_index, transcript):
        self.chunk_segment_indexes = chunk_segment_indexes
        self.transcript = transcript
        self.chunk_index = chunk_index

    def __repr__(self):
        return f"TranscriptChunk(chunk_segment_indexes={self.chunk_segment_indexes}, chunk_index={self.chunk_index})"

    @property
    def length_seconds(self):
        return self.segments[-1].end - self.segments[0].start

    @property
    def segments(self):
        return [self.transcript.segments[i] for i in self.chunk_segment_indexes]

    # TODO this returns segments, while kept_segments in Transcript (not TranscriptChunk) returns indexes
    @property
    def kept_segments(self):
        return [
            self.transcript.segments[seg_i]
            for seg_i in self.chunk_segment_indexes
            if seg_i in self.transcript.kept_segments
        ]

    @property
    def kept_segment_indexes(self):
        return [
            seg_i
            for seg_i in self.chunk_segment_indexes
            if seg_i in self.transcript.kept_segments
        ]

    @property
    def kept_words(self):
        return [w for segment in self.kept_segments for w in segment.kept_words]

    @property
    def words(self):
        return [w for segment in self.segments for w in segment.words]

    @property
    def kept_word_count(self):
        return len(self.kept_words)

    @property
    def text(self):
        return " ".join([segment.text for segment in self.kept_segments])

    @property
    def text_with_keep_tags(self):
        words = []
        for i, segment in zip(self.chunk_segment_indexes, self.segments):
            if i not in self.kept_segment_indexes:
                words.extend(segment.words)
                continue
            if segment.words_with_keep_tags:
                words.extend(segment.words_with_keep_tags)
        return " ".join(words)

    @property
    def text_with_keep_and_remove_tags(self):
        words = []
        for i, segment in zip(self.chunk_segment_indexes, self.segments):
            if i not in self.kept_segment_indexes:
                words.extend(["<remove>"] + segment.words + ["</remove>"])
                continue
            if segment.words_with_keep_and_remove_tags:
                words.extend(segment.words_with_keep_and_remove_tags)
        return " ".join(words)

    @property
    def start_offset(self):
        start_segment_i = self.chunk_segment_indexes[0]
        return sum(
            [len(self.transcript.segments[i].words) for i in range(start_segment_i)]
        )

    @property
    def kept_start_offset(self):
        start_segment_i = self.chunk_segment_indexes[0]
        return sum(
            [
                len(self.transcript.segments[i].kept_words)
                for i in range(start_segment_i)
                if i in self.transcript.kept_segments
            ]
        )

    def full_offsets_from_kept_offsets(self, kept_offsets: list[tuple[int, int]]):
        offsets_converted_to_transcript = [
            (s + self.kept_start_offset, e + self.kept_start_offset)
            for s, e in kept_offsets
        ]
        return [
            offset
            for kept_offset in offsets_converted_to_transcript
            for offset in self.transcript.contiguous_full_word_offsets_from_kept_offset(
                kept_offset
            )
        ]

    def seg_offsets_from_text_offsets(self, text_offsets: list[tuple[int, int]]):
        offsets_converted_to_transcript = [
            (s + self.start_offset, e + self.start_offset) for s, e in text_offsets
        ]
        return [
            self.transcript.seg_offset_from_text_offset(text_offset)
            for text_offset in offsets_converted_to_transcript
        ]

    def copy(self):
        transcript = self.transcript.copy()
        return transcript.chunks[self.chunk_index]

    def erase_cuts(self):
        self.transcript.erase_cuts(self.chunk_segment_indexes)

    def keep_only_cuts(self, offsets: list[OffsetToCut]):
        # Assumes offsets are full offsets from transcript, not chunk
        self.transcript.keep_only_cuts(offsets, from_chunk=self)

    def full_segment_index_from_chunk_segment_index(self, chunk_segment_index: int):
        return self.chunk_segment_indexes[0] + chunk_segment_index

    @property
    def state(self):
        # This state is used by Transcript and is different from the one we used to just save a chunk
        return {
            "chunk_segment_indexes": self.chunk_segment_indexes,
            "chunk_index": self.chunk_index,
        }

    def save(self, file):
        with open(file, "wb") as f:
            transcript_state = self.transcript.state
            # Since chunks store a full copy of the transcript,
            # the only additional information we need to save besides the transcript's state
            # is the chunk index
            state_to_save = {
                "chunk_index": self.chunk_index,
                "transcript_state": transcript_state,
            }
            pickle.dump(state_to_save, f)

    @classmethod
    def load_from_file(cls, file):
        with open(file, "rb") as f:
            state = pickle.load(f)
        return cls.load_from_state(state)

    @classmethod
    def load_from_state(cls, state):
        transcript = Transcript.load_from_state(state["transcript_state"])
        return transcript.chunks[state["chunk_index"]]


class Transcript:
    def __init__(
        self,
        segments: list[TranscriptSegment],
        kept_segments=None,
        partition_boundaries=None,
        chunks=None,
    ):
        self.segments = segments
        self.kept_segments = kept_segments or set(range(len(segments)))
        self.partition_boundaries = partition_boundaries or []
        self.chunks = chunks or []

    @property
    def length_seconds(self):
        return self.segments[-1].end - self.segments[0].start

    def remove_segment(self, index):
        self.kept_segments.remove(index)

    def cut_segment(self, index, start_index, end_index):
        if index not in self.kept_segments:
            raise ValueError("Segment has already been removed")
        self.segments[index].cut(start_index, end_index)

    @property
    def kept_segments_list(self):
        return sorted(list(self.kept_segments))

    def _create_partition_boundary(self, segment_index, word_index):
        # special case for 0,0
        if (
            segment_index > 0
            and word_index > 0
            and segment_index not in self.kept_segments
        ):
            raise ValueError("Chunk boundary segment has already been removed")
        segment = self.segments[segment_index]
        if word_index >= len(segment.words):
            raise ValueError("Word index out of range")
        # only add a split if word_index is in the middle of the segment
        if word_index > 0 and word_index < len(segment.words) - 1:
            new_left_segment, new_right_segment = segment.split(word_index)
            new_segments = (
                self.segments[:segment_index]
                + [new_left_segment, new_right_segment]
                + self.segments[segment_index + 1 :]
            )
            new_kept_segments = set(
                [seg_i for seg_i in self.kept_segments if seg_i <= segment_index]
            )
            # add the new split segment
            new_kept_segments.add(segment_index + 1)
            # bump the segments to the right since we added one
            new_kept_segments_right = set(
                [seg_i + 1 for seg_i in self.kept_segments if seg_i > segment_index]
            )
            new_kept_segments = new_kept_segments.union(new_kept_segments_right)
            self.segments = new_segments
            self.kept_segments = new_kept_segments

        partition_boundary = TranscriptPartitionBoundary(
            segment_index=segment_index, word_index=word_index
        )
        self.partition_boundaries.append(partition_boundary)

    def _create_chunks_from_partition_boundaries(self):
        self.chunks = []
        assert len(self.partition_boundaries) > 0, "No partition boundaries created"
        for partition_index in range(len(self.partition_boundaries) - 1):
            chunk_start_partition = self.partition_boundaries[partition_index]
            chunk_end_partition = self.partition_boundaries[partition_index + 1]
            chunk_segment_indexes = [
                seg_i
                for seg_i in range(len(self.segments))
                if seg_i >= chunk_start_partition.segment_index
                and seg_i < chunk_end_partition.segment_index
            ]
            self.chunks.append(
                TranscriptChunk(chunk_segment_indexes, partition_index, self)
            )
        last_chunk_segment_indexes = [
            seg_i
            for seg_i in range(len(self.segments))
            if seg_i >= self.partition_boundaries[-1].segment_index
        ]
        self.chunks.append(
            TranscriptChunk(
                last_chunk_segment_indexes, len(self.partition_boundaries) - 1, self
            )
        )

    @property
    def text(self):
        return " ".join(
            [
                segment.text
                for i, segment in enumerate(self.segments)
                if i in self.kept_segments and segment.text
            ]
        )

    @property
    def text_with_keep_tags(self):
        words = []
        for i, segment in enumerate(self.segments):
            if i not in self.kept_segments:
                words.extend(segment.words)
                continue
            if segment.words_with_keep_tags:
                words.extend(segment.words_with_keep_tags)
        return " ".join(words)

    @property
    def text_with_keep_and_remove_tags(self):
        words = []
        for i, segment in enumerate(self.segments):
            if i not in self.kept_segments:
                words.extend(["<remove>"] + segment.words + ["</remove>"])
                continue
            segment_words_with_keep_and_remove_tags = (
                segment.words_with_keep_and_remove_tags
            )
            if segment_words_with_keep_and_remove_tags:
                words.extend(segment_words_with_keep_and_remove_tags)
        return " ".join(words)

    @property
    def text_with_speaker_tags(self):
        parsed = []
        for segment in self.segments:
            speaker = segment.speaker or "speaker_UNKNOWN"
            if not speaker.lower().startswith("speaker_"):
                speaker = f"speaker_{speaker}"
            parsed_segment = f"<{speaker.lower()}>{segment.text}</{speaker.lower()}>"
            parsed.append(parsed_segment)
        return "".join(parsed)

    @property
    def kept_words(self):
        return [
            w
            for i, segment in enumerate(self.segments)
            for w in segment.kept_words
            if i in self.kept_segments
        ]

    @property
    def kept_word_count(self):
        return len(self.kept_words)

    @property
    def words(self):
        return [w for segment in self.segments for w in segment.words]

    @property
    def word_count(self):
        return len(self.words)

    def iter_cuts(self):
        for i, segment in enumerate(self.segments):
            for cut in segment.iter_cuts():
                cut.is_kept = cut.is_kept if i in self.kept_segments else False
                yield cut

    def iter_kept_cuts(self):
        for cut in self.iter_cuts():
            if cut.is_kept:
                yield cut

    @classmethod
    def from_video_transcription(cls, transcription: dict):
        transcript_segments = [
            TranscriptSegment.from_video_transcription_segment(segment)
            for segment in transcription["segments"]
        ]
        return cls(transcript_segments)

    def copy(self):
        transcript_copy = Transcript(
            segments=[s.copy() for s in self.segments],
            kept_segments=copy.deepcopy(self.kept_segments),
            partition_boundaries=[pb.copy() for pb in self.partition_boundaries],
            chunks=None,
        )

        transcript_copy.chunks = [
            TranscriptChunk(
                chunk_segment_indexes=chunk.chunk_segment_indexes,
                transcript=transcript_copy,
                chunk_index=chunk.chunk_index,
            )
            for chunk in self.chunks
        ]
        return transcript_copy

    def split_in_chunks(self, chunk_length):
        # TODO split on commas too
        self.partition_boundaries = []
        word_index_to_segment_index = {}
        kept_words = 0
        periods = []
        full_word_i = 0
        for seg_i, segment in enumerate(self.segments):
            if seg_i not in self.kept_segments:
                continue
            for word_i, word in enumerate(segment.words):
                if segment.word_i_kept(word_i):
                    kept_words += 1
                    if word.strip().endswith("."):
                        word_index_to_segment_index[full_word_i] = (seg_i, word_i)
                        periods.append(full_word_i)
                full_word_i += 1

        chunks = range(0, kept_words, chunk_length)
        closest_periods = [-1]
        available_periods = periods[:]
        if not available_periods:
            available_periods = list(chunks)
        for i in chunks:
            if len(available_periods) == 0:
                available_chunks = [c for c in chunks if c > closest_periods[-1]]
                if len(available_chunks) == 0:
                    break
                nearest_chunk = min(available_chunks, key=lambda x: abs(x - i))
                closest_periods.append(nearest_chunk)
                continue
            nearest_next_period = min(
                available_periods, key=lambda x: abs(x - (i + chunk_length))
            )
            available_periods = available_periods[
                available_periods.index(nearest_next_period) + 1 :
            ]
            closest_periods.append(nearest_next_period)
        # TODO this code is garbage. Should refer to each segment by hash/key or something
        self._create_partition_boundary(0, 0)
        n_added = 0
        for i, period in enumerate(closest_periods[1:-1]):
            seg_index, word_index = word_index_to_segment_index[period]
            prev_n_segments = len(self.segments)
            self._create_partition_boundary(seg_index + n_added, word_index)
            n_added += len(self.segments) - prev_n_segments
        self._create_chunks_from_partition_boundaries()
        return self.chunks

    def full_offsets_from_kept_offsets(self, kept_offsets: list[tuple[int, int]]):
        return [
            offset
            for kept_offset in kept_offsets
            for offset in self.contiguous_full_word_offsets_from_kept_offset(
                kept_offset
            )
        ]

    def seg_offsets_from_text_offsets(self, text_offsets: list[tuple[int, int]]):
        return [
            self.seg_offset_from_text_offset(text_offset)
            for text_offset in text_offsets
        ]

    def seg_offset_from_text_offset(self, offset: tuple[int, int]):
        full_offset_start = None
        full_offset_end = None
        word_index = 0
        for seg_i, segment in enumerate(self.segments):
            for word_offset in range(len(segment.words)):
                if word_index == offset[0]:
                    full_offset_start = (seg_i, word_offset)
                elif word_index == offset[1]:
                    if word_offset == 0:
                        if seg_i == 0:
                            raise ValueError(f"offset {offset} out of range")
                        full_offset_end = (
                            seg_i - 1,
                            len(self.segments[seg_i - 1].words),
                        )
                    else:
                        full_offset_end = (seg_i, word_offset)
                word_index += 1
        if full_offset_start is None:
            raise ValueError(f"start offset {offset} out of range")
        if full_offset_end is None:
            full_offset_end = (len(self.segments) - 1, len(self.segments[-1].words))
        return OffsetToCut(
            seg_i_start=full_offset_start[0],
            word_i_start=full_offset_start[1],
            seg_i_end=full_offset_end[0],
            word_i_end=full_offset_end[1],
        )

    def contiguous_full_word_offsets_from_kept_offset(
        self, kept_offset: tuple[int, int]
    ):
        contiguous_full_offsets = []
        full_kept_word_index = 0
        in_contiguous = False
        for seg_i, segment in enumerate(self.segments):
            segment_kept_word_offset = 0
            for word_offset in range(len(segment.words)):
                if seg_i in self.kept_segments and segment.word_i_kept(word_offset):
                    # found start
                    if (
                        full_kept_word_index + segment_kept_word_offset
                        == kept_offset[0]
                    ):
                        in_contiguous = True
                        start = (seg_i, word_offset)
                    # found end
                    elif (
                        full_kept_word_index + segment_kept_word_offset
                        == kept_offset[1]
                    ):
                        # in this case, we have only seen non-kept words since the last offset (start has not changed)
                        # so we the fact that now we're at kept_offset[1] is meaningless- it's just because we use exclusive-intervals on the right
                        # so can ignore everything so far and just return
                        if len(contiguous_full_offsets) and (start[0], start[1]) == (
                            contiguous_full_offsets[-1].seg_i_start,
                            contiguous_full_offsets[-1].word_i_start,
                        ):
                            return contiguous_full_offsets
                        contiguous_full_offsets.append(
                            OffsetToCut(
                                seg_i_start=start[0],
                                word_i_start=start[1],
                                seg_i_end=seg_i,
                                word_i_end=word_offset,
                            )
                        )
                        return contiguous_full_offsets
                    # found new contiguous inner segment
                    elif (
                        not in_contiguous
                        and full_kept_word_index + segment_kept_word_offset
                        > kept_offset[0]
                    ):
                        start = (seg_i, word_offset)
                        in_contiguous = True
                    segment_kept_word_offset += 1
                # found non contiguous inner word: end the current segment
                elif in_contiguous:
                    contiguous_full_offsets.append(
                        OffsetToCut(
                            seg_i_start=start[0],
                            word_i_start=start[1],
                            seg_i_end=seg_i,
                            word_i_end=word_offset,
                        )
                    )
                    in_contiguous = False
            full_kept_word_index += segment_kept_word_offset
        # TODo double check this is correct
        # last word is final length of segment
        if in_contiguous and (start[0], start[1]) != (seg_i, word_offset):
            contiguous_full_offsets.append(
                OffsetToCut(
                    seg_i_start=start[0],
                    word_i_start=start[1],
                    seg_i_end=seg_i,
                    word_i_end=word_offset + 1,
                )
            )
            return contiguous_full_offsets
        else:
            return contiguous_full_offsets
        # raise ValueError(f"end kept offset {kept_offset} out of range")

    def erase_cuts(self, segment_indexes=None):
        if segment_indexes is None:
            self.kept_segments = set()
        else:
            self.kept_segments = self.kept_segments.difference(set(segment_indexes))

        segments_to_erase = [
            seg
            for seg_i, seg in enumerate(self.segments)
            if segment_indexes is None or seg_i in segment_indexes
        ]
        for seg in segments_to_erase:
            seg.cut_segments = []

    def restrict_offset_to_chunk(self, offset: OffsetToCut, chunk: TranscriptChunk):
        new_offset = offset.model_copy()
        if offset.seg_i_start < chunk.chunk_segment_indexes[0]:
            new_offset.seg_i_start = chunk.chunk_segment_indexes[0]
            new_offset.word_i_start = 0
        if offset.seg_i_end < chunk.chunk_segment_indexes[0]:
            return None
        if offset.seg_i_start > chunk.chunk_segment_indexes[-1]:
            return None
        if offset.seg_i_end > chunk.chunk_segment_indexes[-1]:
            if offset.word_i_end > 0:
                new_offset.seg_i_end = chunk.chunk_segment_indexes[-1] + 1
                new_offset.word_i_end = 0
        assert new_offset.seg_i_end > new_offset.seg_i_start or (
            new_offset.seg_i_end == new_offset.seg_i_start
            and new_offset.word_i_end > new_offset.word_i_start
        )
        return new_offset

    def keep_only_cuts(
        self, offsets: list[OffsetToCut], from_chunk: TranscriptChunk | None = None
    ):
        from trimit.backend.utils import linearize_and_dedupe_offsets

        segment_indexes_to_erase = None
        if from_chunk is not None:
            segment_indexes_to_erase = from_chunk.chunk_segment_indexes
        self.erase_cuts(segment_indexes_to_erase)
        # TODO allow reordering and reuse of cuts!
        offsets = linearize_and_dedupe_offsets(offsets)

        for offset_to_cut in offsets:
            if from_chunk is not None:
                offset_to_cut = self.restrict_offset_to_chunk(offset_to_cut, from_chunk)
                if offset_to_cut is None:
                    print(
                        f"Offset {offset_to_cut} not in chunk {from_chunk.chunk_index}. Skipping"
                    )
                    continue
            seg_i_start, word_i_start, seg_i_end, word_i_end = (
                offset_to_cut.seg_i_start,
                offset_to_cut.word_i_start,
                offset_to_cut.seg_i_end,
                offset_to_cut.word_i_end,
            )
            self.kept_segments = self.kept_segments.union(
                set(range(seg_i_start, seg_i_end + 1))
            )
            seg_start = self.segments[seg_i_start]
            if seg_i_end == seg_i_start:
                if word_i_end == word_i_start:
                    continue
                seg_start.cut(word_i_start, word_i_end)
            else:
                seg_start.cut(word_i_start, len(seg_start.words))
                if word_i_end > 0:
                    seg_end = self.segments[seg_i_end]
                    seg_end.cut(0, word_i_end)
                if seg_i_end > seg_i_start + 1:
                    for seg_i in range(seg_i_start + 1, seg_i_end):
                        self.segments[seg_i].cut(0, len(self.segments[seg_i].words))

    def cut_from_chunk(self, chunk: TranscriptChunk):
        # assumes chunk comes from different (copied) Transcript object
        internal_chunk = self.chunks[chunk.chunk_index]
        self.kept_segments = set(
            [
                i
                for i in self.kept_segments
                if i not in internal_chunk.chunk_segment_indexes
            ]
        )
        self.kept_segments = self.kept_segments.union(set(chunk.kept_segment_indexes))
        for segment_index in chunk.kept_segment_indexes:
            chunk_segment = chunk.transcript.segments[segment_index]
            segment = self.segments[segment_index]
            segment.cut_segments = chunk_segment.cut_segments

    @classmethod
    def merge(cls, *transcripts: TranscriptChunk):
        # TODO really need to fix kept_segment difference between TranscriptChunk and Transcript
        # Assumes all transcripts have the same segments
        if isinstance(transcripts[0], Transcript):
            raise NotImplementedError("Merging transcripts not yet implemented")
        merged = transcripts[0].copy()
        merged.transcript.kept_segments = set(merged.kept_segment_indexes)
        for other in transcripts[1:]:
            for seg_idx, segment in enumerate(other.transcript.segments):
                if seg_idx in other.kept_segment_indexes:
                    merged.transcript.kept_segments.add(seg_idx)
                    merged.transcript.segments[seg_idx].union_with(segment)
        return merged.transcript

    @property
    def state(self):
        return {
            "segments": [seg.state for seg in self.segments],
            "kept_segments": self.kept_segments,
            "partition_boundaries": self.partition_boundaries,
            "chunks": [chunk.state for chunk in self.chunks],
        }

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.state, f)

    @classmethod
    def load_from_file(cls, file):
        with open(file, "rb") as f:
            state = pickle.load(f)
        return cls.load_from_state(state)

    @classmethod
    def load_from_state(cls, state):
        segments = [TranscriptSegment.load_from_state(seg) for seg in state["segments"]]
        transcript = cls(
            segments=segments,
            kept_segments=set(state["kept_segments"]),
            partition_boundaries=state["partition_boundaries"],
        )
        transcript.chunks = [
            TranscriptChunk(transcript=transcript, **chunk_state)
            for chunk_state in state["chunks"]
        ]
        return transcript


class Soundbite(BaseModel):
    # soundbites must be full segments for now
    start_segment_index: int
    start_word_index: Optional[int] = 0
    end_segment_index: int
    end_word_index: Optional[int] = None
    _soundbites_obj: "Soundbites" = None

    @classmethod
    def from_dict_with_parent_obj(
        cls,
        parent_obj,
        start_segment_index,
        end_segment_index,
        start_word_index=0,
        end_word_index=None,
    ):
        sb = cls(
            start_segment_index=start_segment_index,
            end_segment_index=end_segment_index,
            start_word_index=start_word_index,
            end_word_index=end_word_index,
        )
        sb._soundbites_obj = parent_obj
        return sb

    @property
    def soundbite_segments(self):
        if not hasattr(self, "_soundbites_obj"):
            raise AttributeError(
                "must set _soundbites_obj to get segments directly from Soundbite instance"
            )
        transcript = self._soundbites_obj.transcript
        return [
            transcript.segments[i]
            for i in range(self.start_segment_index, self.end_segment_index + 1)
            if i in transcript.kept_segments
        ]

    def iter_kept_cuts(self):
        segments = self.soundbite_segments
        if len(segments) == 1:
            for cut in segments[0].kept_cuts_with_start_end(
                self.start_word_index, self.end_word_index
            ):
                yield cut
        elif len(segments) >= 2:
            for cut in segments[0].kept_cuts_with_start_end(
                self.start_word_index, end_word_index=None
            ):
                yield cut
            if len(segments) > 2:
                for seg in segments[1:-1]:
                    for cut in seg.iter_cuts():
                        yield cut
            for cut in segments[-1].kept_cuts_with_start_end(
                end_word_index=self.end_word_index
            ):
                yield cut


class SoundbitesChunk(TranscriptChunk):
    def __init__(self, chunk_segment_indexes, chunk_index, soundbites_object):
        super().__init__(
            chunk_segment_indexes, chunk_index, soundbites_object.transcript
        )
        self.soundbites_object = soundbites_object

    def __repr__(self):
        return f"SoundbitesChunk(chunk_segment_indexes={self.chunk_segment_indexes}, chunk_index={self.chunk_index})"

    @classmethod
    async def from_keep_tags(cls, transcript_chunk, text_with_keep_tags):
        soundbites = await Soundbites.from_keep_tags(
            transcript_chunk.transcript, text_with_keep_tags
        )
        return soundbites.chunks[transcript_chunk.chunk_index]

    def iter(self):
        for i, soundbite in self.soundbites_object.iter():
            if soundbite.start_segment_index not in self.chunk_segment_indexes:
                continue
            yield i, soundbite

    @property
    def soundbites_indexes(self):
        # TODO make this more efficient
        return set([i for i, _ in self.iter()])

    def iter_text(self):
        soundbites_indexes = self.soundbites_indexes
        return [
            (i, soundbite)
            for i, soundbite in self.soundbites_object.iter_text()
            if i in soundbites_indexes
        ]

    def iter_text_list(self):
        return [(i, text) for i, text in self.iter_text()]

    @property
    def soundbites(self):
        return [
            sb
            for sb in self.soundbites_object.soundbites
            if sb.start_segment_index >= self.chunk_segment_indexes[0]
            and sb.end_segment_index <= self.chunk_segment_indexes[-1]
        ]

    def remove_existing_chunk_soundbites(self):
        self.soundbites_object.soundbites = [
            sb
            for sb in self.soundbites_object.soundbites
            if sb.start_segment_index < self.chunk_segment_indexes[0]
            or sb.end_segment_index > self.chunk_segment_indexes[-1]
        ]

    @soundbites.setter
    def soundbites(self, new_chunk_soundbites):
        self.remove_existing_chunk_soundbites()
        for soundbite in new_chunk_soundbites:
            if isinstance(soundbite, Soundbite):
                soundbite = soundbite.model_dump()
            self.soundbites_object.add_soundbite(**soundbite)

    def keep_only_in_transcript(self, transcript: Transcript | TranscriptChunk):
        if isinstance(transcript, TranscriptChunk):
            return self.keep_only_in_transcript_chunk(transcript)
        new_instance = self.copy()
        new_instance.soundbites = [
            soundbite
            for soundbite in self.soundbites
            if soundbite.start_segment_index in transcript.kept_segments
        ]
        return new_instance

    def keep_only_in_transcript_chunk(self, transcript_chunk: TranscriptChunk):
        new_instance = self.copy()
        new_instance.soundbites = [
            soundbite
            for soundbite in self.soundbites
            if soundbite.start_segment_index in transcript_chunk.kept_segment_indexes
        ]
        return new_instance

    def copy(self):
        soundbites_object = self.soundbites_object.copy()
        return soundbites_object.chunks[self.chunk_index]

    def save(self, file):
        with open(file, "wb") as f:
            soundbites_state = self.soundbites_object.state
            state_to_save = {
                "chunk_index": self.chunk_index,
                "soundbites_state": soundbites_state,
            }
            pickle.dump(state_to_save, f)

    @classmethod
    def load_from_state(cls, state):
        soundbites = Soundbites.load_from_state(state["soundbites_state"])
        return soundbites.chunks[state["chunk_index"]]


class Soundbites(Transcript):
    def __init__(self, transcript, soundbites: list[Soundbite | dict]):
        self.transcript = transcript
        self.soundbites = []
        for soundbite in soundbites:
            if isinstance(soundbite, Soundbite):
                soundbite = soundbite.model_dump()
            self.add_soundbite(**soundbite)

    @property
    def chunks(self):
        return [
            SoundbitesChunk(
                chunk_segment_indexes=chunk.chunk_segment_indexes,
                chunk_index=chunk.chunk_index,
                soundbites_object=self,
            )
            for chunk in self.transcript.chunks
        ]

    def iter_soundbite_segments(self):
        for i in range(len(self.soundbites)):
            yield i, self.soundbites[i].soundbite_segments

    def iter(self):
        return enumerate(self.soundbites)

    def iter_text_list(self):
        return [(i, text) for i, text in self.iter_text()]

    def iter_text(self):
        for i, segments in self.iter_soundbite_segments():
            soundbite = self.soundbites[i]
            if len(segments) == 1:
                yield i, segments[0].text_with_start_end(
                    soundbite.start_word_index, soundbite.end_word_index
                )
            elif len(segments) >= 2:
                text_segs = [
                    segments[0].text_with_start_end(
                        soundbite.start_word_index, end_word_index=None
                    )
                ]
                if len(segments) > 2:
                    text_segs.extend([seg.text for seg in segments[1:-1]])
                text_segs.append(
                    segments[-1].text_with_start_end(
                        end_word_index=soundbite.end_word_index
                    )
                )
                yield i, " ".join([s for s in text_segs if s])

    def iter_kept_cuts(self):
        for soundbite in self.soundbites:
            for cut in soundbite.iter_kept_cuts():
                yield cut

    @property
    def text(self):
        return "\n".join([seg_text for i, seg_text in self.iter_text() if seg_text])

    def copy(self):
        return Soundbites(self.transcript.copy(), [s.copy() for s in self.soundbites])

    @classmethod
    async def from_keep_tags(cls, transcript, text_with_keep_tags):
        from trimit.backend.utils import match_output_to_actual_transcript_fast

        transcript, offsets = match_output_to_actual_transcript_fast(
            transcript, text_with_keep_tags, return_offsets=True
        )
        soundbites_obj = cls(transcript, [])
        for offset in offsets:
            soundbites_obj.add_soundbite(
                start_segment_index=offset.seg_i_start,
                start_word_index=offset.word_i_start,
                end_segment_index=offset.seg_i_end,
                end_word_index=offset.word_i_end,
            )
        return soundbites_obj

    def keep_only_in_transcript(self, transcript: Transcript | TranscriptChunk):
        if isinstance(transcript, TranscriptChunk):
            return self.keep_only_in_transcript_chunk(transcript)
        new_instance = self.copy()
        new_instance.soundbites = [
            soundbite
            for soundbite in self.soundbites
            if soundbite.start_segment_index in transcript.kept_segments
        ]
        return new_instance

    def keep_only_in_transcript_chunk(self, transcript_chunk: TranscriptChunk):
        new_instance = self.copy()
        new_instance.soundbites = [
            soundbite
            for soundbite in self.soundbites
            if soundbite.start_segment_index in transcript_chunk.kept_segment_indexes
        ]
        return new_instance

    def add_soundbite(
        self,
        start_segment_index,
        end_segment_index,
        start_word_index=0,
        end_word_index=None,
    ):
        soundbite = Soundbite.from_dict_with_parent_obj(
            parent_obj=self,
            start_segment_index=start_segment_index,
            end_segment_index=end_segment_index,
            start_word_index=start_word_index,
            end_word_index=end_word_index,
        )
        assert soundbite.start_segment_index in self.transcript.kept_segments
        assert soundbite.end_segment_index in self.transcript.kept_segments
        start_seg = self.transcript.segments[soundbite.start_segment_index]
        start_seg.cut_segments = [
            cut_seg
            for cut_seg in start_seg.cut_segments
            if cut_seg[0] < soundbite.start_word_index
        ]
        if (
            start_seg.cut_segments
            and start_seg.cut_segments[-1][-1] >= soundbite.start_word_index
        ):
            last_cut_seg = start_seg.cut_segments[-1]
            last_cut_seg = (last_cut_seg[0], soundbite.start_word_index - 1)
            start_seg.cut_segments[-1] = last_cut_seg
        start_seg.cut(soundbite.start_word_index, len(start_seg.words))

        end_seg = self.transcript.segments[soundbite.end_segment_index]
        # TODO need better interface to cut
        end_seg.cut_segments = [
            cut_seg
            for cut_seg in end_seg.cut_segments
            if cut_seg[-1] >= soundbite.end_word_index
        ]
        if (
            end_seg.cut_segments
            and end_seg.cut_segments[0][0] < soundbite.end_word_index
        ):
            first_cut_seg = end_seg.cut_segments[0]
            first_cut_seg = (soundbite.end_word_index, first_cut_seg[1])
            end_seg.cut_segments[0] = first_cut_seg
        end_seg.cut(0, soundbite.end_word_index)

        self.soundbites.append(soundbite)
        self.soundbites = sorted(
            self.soundbites, key=lambda x: (x.start_segment_index, x.start_word_index)
        )

    @classmethod
    def merge(cls, *soundbites_chunks: SoundbitesChunk):
        all_soundbites = [s for chunk in soundbites_chunks for s in chunk.soundbites]
        merged = Transcript.merge(*[chunk for chunk in soundbites_chunks])
        return cls(merged, all_soundbites)

    def align_to_transcript_chunks(self, transcript):
        self.transcript.chunks = transcript.chunks
        self.transcript.partition_boundaries = transcript.partition_boundaries

    @property
    def state(self):
        return {"transcript": self.transcript.state, "soundbites": self.soundbites}

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.state, f)

    @classmethod
    def load_from_file(cls, file):
        with open(file, "rb") as f:
            state = pickle.load(f)
        return cls.load_from_state(state)

    @classmethod
    def load_from_state(cls, state):
        transcript = Transcript.load_from_state(state["transcript"])
        return cls(transcript=transcript, soundbites=state["soundbites"])


class PartialFeedback(BaseModel):
    partials_to_redo: list[bool] = Field(
        ...,
        description="list of bools, length of number of partial chunks of transcript, indicate which chunks to redo",
    )
    relevant_user_feedback_list: list[str | None] | list[str] = Field(
        ..., description="specific feedback provided by LLM to each individual chunk"
    )


class CutTranscriptLinearWorkflowStepInput(BaseModel):
    user_prompt: str | None = Field(
        None, description="prompt provided by user to this step"
    )
    llm_modified_partial_feedback: PartialFeedback | None = Field(
        None,
        description="feedback provided by LLM after processing the raw user_prompt",
    )
    is_retry: bool = Field(False, description="whether the current run is a retry")
    step_name: str | None = None
    substep_name: str | None = None


class CutTranscriptLinearWorkflowStepResults(BaseModel):
    user_feedback_request: str | None = None
    retry: bool = False
    outputs: dict | None = None


class CutTranscriptLinearWorkflowStepOutput(BaseModel):
    step_name: str
    substep_name: str
    done: bool = False
    user_feedback_request: str | None = Field(
        None,
        description="prompt to user for additional, optional input if the user wants to rerun the step",
    )
    partial_user_feedback_request: str | None = None
    step_inputs: CutTranscriptLinearWorkflowStepInput | None = Field(
        None,
        description="the inputs that were provided to this step when it was most recently run",
    )
    step_outputs: dict | None = Field(
        None,
        description="the produced outputs from this step that were saved to the workflow's state",
    )
    export_result: dict[str, str | list[str] | dict[str, list[str]]] | None = Field(
        None,
        description="the files produced by this step. May be None if files are not ready yet. In that case, poll for them to be ready using export_call_id and then call /get_step_outputs",
    )
    export_call_id: str | None = Field(
        None,
        description="backend call_id that can be polled to wait until export files are fully created",
    )
    error: str | None = None
    retry: bool = Field(
        False,
        description="frontend should ignore this. backend may choose to retry the current step before asking the user for feedback, in which case this will be true",
    )

    def merge(self, other: "CutTranscriptLinearWorkflowStepOutput"):
        if self.step_name == "" and other.step_name:
            self.step_name = other.step_name
        elif self.step_name and other.step_name == "":
            other.step_name = self.step_name
        elif self.step_name and other.step_name and self.step_name != other.step_name:
            raise ValueError(
                f"Cannot merge outputs of different steps: {self.step_name} and {other.step_name}"
            )

        if self.substep_name == "" and other.substep_name:
            self.substep_name = other.substep_name
        elif self.substep_name and other.substep_name == "":
            other.substep_name = self.substep_name
        elif (
            self.substep_name
            and other.substep_name
            and self.substep_name != other.substep_name
        ):
            raise ValueError(
                f"Cannot merge outputs of different steps: {self.substep_name} and {other.substep_name}"
            )
        self.done = self.done or other.done
        self.user_feedback_request = (
            self.user_feedback_request or other.user_feedback_request
        )
        self.partial_user_feedback_request = (
            self.partial_user_feedback_request or other.partial_user_feedback_request
        )
        self.step_inputs = self.step_inputs or other.step_inputs
        self.step_outputs = self.step_outputs or other.step_outputs
        self.export_result = self.export_result or other.export_result
        self.export_call_id = self.export_call_id or other.export_call_id
        self.error = self.error or other.error
        self.retry = self.retry or other.retry


class CurrentStepInfo(BaseModel):
    name: str
    user_feedback: bool
    method: Callable | None = None
    input: CutTranscriptLinearWorkflowStepInput | None = None
    chunked_feedback: bool = False
    step: Union["StepWrapper", None] = None
    step_name: str | None = None
    export_transcript: Optional[bool] = True
    export_video: Optional[bool] = True
    export_soundbites: Optional[bool] = True
    export_timeline: Optional[bool] = True
    export_speaker_tagging: Optional[bool] = False

    def model_dump_json(self, *args, **kwargs):
        return json.dumps(self.model_dump(*args, **kwargs))

    def model_dump(self, *args, exclude=None, **kwargs):
        step_name_in_exclude = False
        if exclude is None:
            exclude = {"method", "step", "step_name"}
        else:
            step_name_in_exclude = "step_name" in exclude
            exclude |= {"method", "step", "step_name"}
        res = super().model_dump(*args, exclude=exclude, **kwargs)
        if not step_name_in_exclude:
            res["step_name"] = self.step.name if self.step else None
        return res

    def to_dict(self):
        return {
            "name": self.name,
            "user_feedback": self.user_feedback,
            "input": self.input.model_dump() if self.input else None,
            "chunked_feedback": self.chunked_feedback,
            "step_name": self.step.name if self.step else None,
        }

    def to_exportable(self):
        return ExportableStepInfo(
            substep_name=self.name,
            step_name=self.step.name if self.step else "",
            **self.model_dump(exclude={"step_name"}),
        )


class ExportableStepInfo(BaseModel):
    substep_name: str = Field(..., description="name of the substep")
    user_feedback: bool = Field(
        ...,
        description="If True, this step will stop the workflow when finished in order to gather feedback from the user",
    )
    step_name: str = Field(..., description="name of the higher-level step")
    input: CutTranscriptLinearWorkflowStepInput | None = Field(
        None, description="Input that was (or will be) provided to the step's method"
    )
    chunked_feedback: bool = Field(
        False,
        description="If True, this step desires feedback from the user about several independent 'chunks' of its output. For instance, feedback about each soundbite produced.",
    )
    export_transcript: Optional[bool] = Field(
        True,
        description="If True, this step will export a transcript pickle (TODO: modify to export JSON) and a plaintext file",
    )
    export_video: Optional[bool] = Field(
        True, description="If True, this step will export an mp4 video file"
    )
    export_soundbites: Optional[bool] = Field(
        True,
        description="If True, this step will export a transcript (pickle- eventually JSON, and plaintext) of the workflow's soundbites, along with timeline XMLs and video mp4s for each",
    )
    export_timeline: Optional[bool] = Field(
        True, description="If True, this step will export a timeline XML file"
    )
    export_speaker_tagging: Optional[bool] = Field(
        False,
        description="If True, this step will export several sample mp4s for each speaker (annotated with the current speaker tags in the export_result dict mapping to filenames)",
    )


class PartialLLMOutput(BaseModel):
    value: str = Field(
        ..., description="string value of a partial output response from the LLM"
    )
    chunk: int | None = Field(
        None,
        description="chunk index if this output was produced concurrently with several independent calls to the LLM",
    )
    calling_method_name: str | None = Field(
        None,
        description="backend method called to produce this output. Only provided if there are multiple methods called in sequence by the step",
    )


class FinalLLMOutput(BaseModel):
    str_value: str | None = None
    json_value: dict | None = None
    chunk: int | None = None
    calling_method_name: str | None = None


class PartialBackendOutput(BaseModel):
    value: str | None = Field(
        None,
        description="Information provided by the backend about what it is currently doing",
    )

    current_substep: ExportableStepInfo | None = Field(
        None,
        description="Current substep that produced this output, provided if there are more than one that will be called before a feedback request",
    )
    chunk: int | None = Field(
        None, description="chunk index if this output was produced concurrently"
    )


class CutTranscriptLinearWorkflowStreamingOutput(BaseModel):
    # TODO annotate these
    partial_llm_output: PartialLLMOutput | None = Field(
        None, description="Chunk of output from the LLM"
    )
    final_llm_output: FinalLLMOutput | None = Field(
        None, description="Full output from the LLM, not currently send to frontend"
    )
    partial_backend_output: PartialBackendOutput | None = Field(
        None, description="Text output with metadata from the backend"
    )
    partial_step_output: CutTranscriptLinearWorkflowStepOutput | None = Field(
        None,
        description="An output of an intermediary substep that did not require user feedback",
    )
    final_step_output: CutTranscriptLinearWorkflowStepOutput | None = Field(
        None, description="The final output from running the step"
    )


class GetStepOutputs(BaseModel):
    outputs: list[CutTranscriptLinearWorkflowStepOutput]


class CallStatus(BaseModel):
    status: str = Field(..., description="'done', 'pending', or 'error'")
    call_id: str | None
    error: str | None = None


class VideoProcessingStatus(CallStatus):
    video_hash: str

    @classmethod
    def from_call_status(cls, status, video_hash):
        return cls(status=status, **status.model_dump())


class GetVideoProcessingStatus(BaseModel):
    statuses: list[VideoProcessingStatus]


class CheckFunctionCallResults(BaseModel):
    statuses: list[CallStatus]


class StepWrapper(BaseModel):
    name: str
    substeps: list[CurrentStepInfo]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for substep in self.substeps:
            substep.step = self

    def model_dump(self, *args, **kwargs):
        return {"name": self.name, "substeps": [s.model_dump() for s in self.substeps]}

    def to_exportable(self):
        return ExportableStepWrapper(
            name=self.name,
            substeps=[substep.to_exportable() for substep in self.substeps],
        )


class ExportableStepWrapper(BaseModel):
    name: str
    substeps: list[ExportableStepInfo]


class Steps(BaseModel):
    steps: list[StepWrapper]

    @model_serializer()
    def model_dump(self, *args, **kwargs):
        return {"steps": [s.model_dump() for s in self.steps]}

    def to_exportable(self):
        return [step.to_exportable() for step in self.steps]

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __len__(self):
        return len(self.steps)

    # TODO next_substep should reference next_step so we can just return next_substep
    # instead of needing both of these
    def substep_for_index(self, step_index, substep_index):
        if step_index >= len(self.steps):
            return None
        if substep_index >= len(self.steps[step_index].substeps):
            return None
        return self.steps[step_index].substeps[substep_index]

    def next_step_index(self, step_index, substep_index):
        if step_index >= len(self.steps):
            raise ValueError("Step out of bounds")
        cur_step = self.steps[step_index]
        if substep_index >= len(cur_step.substeps):
            raise ValueError("Step out of bounds")
        substep_index += 1
        if substep_index >= len(cur_step.substeps):
            substep_index = 0
            step_index += 1
            if step_index >= len(self.steps):
                return None, None
        return step_index, substep_index

    def last_step_index(self, step_index, substep_index):
        if step_index >= len(self.steps):
            return len(self.steps) - 1, len(self.steps[-1].substeps) - 1
        substep_index -= 1
        if substep_index < 0:
            step_index -= 1
            substep_index = len(self.steps[step_index].substeps) - 1
            if step_index < 0:
                return None, None
        return step_index, substep_index


class GetLatestState(BaseModel):
    user_messages: list[str] = Field(
        [], description="list of all messages user has provided"
    )
    step_history_state: list[StepKey] = Field(
        [],
        description="list of every step and substeps run so far, including retry_{i} prefixes on substeps when a step was retried",
    )
    video_id: str | None = Field(
        None,
        description="id of the video's db model. Provide to future workflow calls for faster retrieval over video_hash",
    )
    user_id: str | None = Field(
        None,
        description="id of the user's db model. Provide to future workflow calls for faster retrieval over user_email",
    )
    last_step: ExportableStepInfo | None = Field(
        None,
        description="information about the most recently run (sub)step. provides additional information over what's is provided in step_history_state, like whether the state includes concurrent chunked responses from the LLM",
    )
    next_step: ExportableStepInfo | None = Field(
        None,
        description="information about the next scheduled (sub)step. provides additional information over what's is provided in step_history_state, like whether the state includes concurrent chunked responses from the LLM",
    )
    all_steps: list[ExportableStepWrapper] | None = Field(
        None,
        description="detailed, ordered, information about each step/substep in this workflow",
    )
    output: CutTranscriptLinearWorkflowStepOutput | None = Field(
        None, description="most recent substep output"
    )


class UploadedVideo(BaseModel):
    filename: str
    video_hash: str
    path: str


class UploadVideo(BaseModel):
    processing_call_id: str | None = Field(
        None,
        description="backend call_id- use this to retrieve the status of video processing from /get_video_processing_status",
    )
    video_hashes: list[str] | None = Field(
        None, description="list of hashes to refer to videos by in future"
    )
    result: str = Field("error", description="'success', or 'error'")
    messages: list[str] = Field([], description="error messages")
