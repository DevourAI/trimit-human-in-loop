import type { UseFormReturn } from 'react-hook-form';

import { type StepItem } from '@/components/ui/stepper';

export interface ListWorkflowParams {
  user_email: string;
  video_hashes?: string[];
}
export interface CreateNewWorkflowParams {
  user_email: string;
  video_hash: string;
  timeline_name: string;
  length_seconds: number;
  nstages: number;
}

export interface StepData {
  user_input: string;
  streaming: boolean;
  force_restart: boolean;
  ignore_running_workflows: boolean;
  retry_step?: boolean;
}

export interface StepOutputParams {
  workflow_id: string;
  step_name: string;
  latest_retry?: boolean;
}

export interface GetLatestStateParams {
  workflow_id: string;
  with_output?: boolean;
  wait_until_done_running?: boolean;
  block_until?: boolean;
  timeout?: number;
}

export interface RevertStepParams {
  workflow_id: string;
  to_before_retries: boolean;
}

export interface RevertStepToParams {
  workflow_id;
  string;
  step_name: string;
}

export interface GetUploadedVideoParams {
  user_email: string;
}

export interface PartialFeedback {
  partials_to_redo: Array<boolean> | null;
  relevant_user_feedback_list: Array<string | null> | null;
}

export interface StepInput {
  user_prompt?: string | null;
  llm_modified_partial_feedback?: PartialFeedback | null;
  is_retry?: boolean;
  step_name?: string | null;
}
export interface SubStepInfo {
  name: string;
  user_feedback?: boolean;
  chunked_feedback?: boolean;
  input?: StepInput;
  human_readable_name?: string;
}

export interface StepInfo extends StepItem {
  name: string;
  step_name: string;
  human_readable_name?: string;
  substeps: SubStepInfo[];
}

export interface StepData {
  stepArray: StepInfo[];
}

export interface UserState {
  all_steps: StepInfo[];
  next_step: StepInfo;
  last_step: StepInfo;
  video_id: string;
  user_id: string;
}

export interface UploadVideoParams {
  videoFile: File;
  userEmail: string;
  timelineName: string;
}

export interface DownloadFileParams {
  user_email: string;
  timeline_name: string;
  length_seconds: number;
  video_hash: string;
  step_name?: string;
  substep_name?: string;
}

export type Video = {
  filename: string;
  remoteUrl: string;
  hash: string;
};

export interface OutputComponentProps {
  value: any;
  exportResult: any;
  onSubmit: (formData: StructuredUserInputInput) => void;
  form: UseFormReturn;
}
