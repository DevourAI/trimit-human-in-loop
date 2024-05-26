import { StepInfo, StepData } from './types'
// TODO this should go in a protobuf, shared between python and js
export const stepData: StepData = {
  stepArray: [
    {
      name: "preprocess_video",
      human_readable_name: "Preprocess Video",
      substeps: [
        {
          name: "init_state",
          human_readable_name: "Initializing",
          user_feedback: false,
          chunked_feedback: false,
        },
        {
          name: "remove_off_screen_speakers",
          human_readable_name: "Remove Off-Screen Speakers",
          user_feedback: true,
          chunked_feedback: false,
        },
        {
          name: "preprocess_export_results",
          human_readable_name: "Export Results",
          user_feedback: false,
          chunked_feedback: false,
        },
      ]
    },
    {
      name: "generate_story",
      human_readable_name: "Generate Narrative Story",
      substeps: [
        {
          name: "generate_story",
          human_readable_name: "Generate Narrative Story",
          user_feedback: true,
          chunked_feedback: false,
        },
        {
          name: "generate_story_export_results",
          human_readable_name: "Export Results",
          user_feedback: false,
          chunked_feedback: false,
        },
      ]

    },
    {
      name: "identify_key_soundbites",
      human_readable_name: "Identify Key Selects",
      substeps: [
        {
          name: "identify_key_soundbites",
          human_readable_name: "Identify Key Selects",
          user_feedback: true,
          chunked_feedback: true,
        },
        {
          name: "soundbites_export_results",
          human_readable_name: "Export Results",
          user_feedback: false,
          chunked_feedback: false,
        },

      ]
    },
    {
      name: "stage_0_generate_transcript",
      human_readable_name: "Generate Transcript, Stage 1",
      substeps: [
        {
          name: "stage_0_cut_partial_transcripts_with_critiques",
          human_readable_name: "Cut & Critique Partial Transcripts",
          user_feedback: false,
          chunked_feedback: true,
        },
        {
          name: "stage_0_modify_transcript_holistically",
          human_readable_name: "Modify Transcript Holistically",
          user_feedback: true,
          chunked_feedback: false,
        },
        {
          name: "stage_0_export_results",
          human_readable_name: "Export Results",
          user_feedback: false,
          chunked_feedback: false,
        },
      ]
    },
    {
      name: "stage_1_generate_transcript",
      human_readable_name: "Generate Transcript, Stage 2",
      substeps: [
        {
          name: "stage_1_cut_partial_transcripts_with_critiques",
          human_readable_name: "Cut & Critique Partial Transcripts",
          user_feedback: false,
          chunked_feedback: true,
        },
        {
          name: "stage_1_modify_transcript_holistically",
          human_readable_name: "Modify Transcript Holistically",
          user_feedback: true,
          chunked_feedback: false,
        },
        {
          name: "stage_1_export_results",
          human_readable_name: "Export Results",
          user_feedback: false,
          chunked_feedback: false,
        },

      ]
    },

  ]
}
