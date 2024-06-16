import { StepData, StepInfo } from './types';
// TODO this should go in a protobuf, shared between python and js
export const stepData: StepData = {
  stepArray: [
    {
      name: 'preprocess_video',
      human_readable_name: 'Preprocess Video',
      substeps: [
        {
          name: 'remove_off_screen_speakers',
          human_readable_name: 'Remove Off-Screen Speakers',
          user_feedback: true,
          chunked_feedback: false,
        },
      ],
    },
    {
      name: 'generate_story',
      human_readable_name: 'Generate Narrative Story',
      substeps: [
        {
          name: 'generate_story',
          human_readable_name: 'Generate Narrative Story',
          user_feedback: true,
          chunked_feedback: false,
        },
      ],
    },
    {
      name: 'identify_key_soundbites',
      human_readable_name: 'Identify Key Selects',
      substeps: [
        {
          name: 'identify_key_soundbites',
          human_readable_name: 'Identify Key Selects',
          user_feedback: true,
          chunked_feedback: true,
        },
      ],
    },
    {
      name: 'stage_0_generate_transcript',
      human_readable_name: 'Generate Transcript, Stage 1',
      substeps: [
        {
          name: 'modify_transcript_holistically',
          human_readable_name: 'Modify Transcript Holistically (Stage 1)',
          user_feedback: true,
          chunked_feedback: false,
        },
      ],
    },
    {
      name: 'stage_1_generate_transcript',
      human_readable_name: 'Generate Transcript, Stage 2',
      substeps: [
        {
          name: 'modify_transcript_holistically',
          human_readable_name: 'Modify Transcript Holistically (Stage 2)',
          user_feedback: true,
          chunked_feedback: false,
        },
      ],
    },
  ],
};

function separateActionSteps(): StepInfo[][] {
  const allSteps = stepData.stepArray.flatMap((step) =>
    step.substeps.map((substep) => {
      return {
        step_name: step.name,
        step_human_readable_name: step.human_readable_name,
        substeps: [],
        ...substep,
      };
    })
  );
  const actionSteps = allSteps.filter((step) => step.user_feedback);
  return [allSteps, actionSteps];
}

export const [allSteps, actionSteps] = separateActionSteps();
