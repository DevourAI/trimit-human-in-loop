import {
  type StepItem,
} from "@/components/ui/stepper"

export interface CommonAPIParams {
  user_email: string
  timeline_name: string
  length_seconds: number
  video_hash: string
}

export interface StepParams extends CommonAPIParams {
  user_input: string
  streaming: boolean
  force_restart: boolean
  ignore_running_workflows: boolean

}

export interface StepOutputParams extends CommonAPIParams {
  step_keys: string // comma separated
  latest_retry?: boolean
}

export interface GetLatestStateParams extends CommonAPIParams {
  with_output?: boolean
  wait_until_done_running?: boolean
  block_until?: boolean
  timeout?: number
}

export interface RevertStepParams extends CommonAPIParams {
  to_before_retries: boolean
}

export interface ResetWorkflowParams extends CommonAPIParams {}
export interface GetUploadedVideoParams {
  user_email: string
}


export interface PartialFeedback {
    partials_to_redo: Array<bool> | null
    relevant_user_feedback_list: Array<str | null> | null
}

export interface StepInput {
    user_prompt?: string | null
    llm_modified_partial_feedback?: PartialFeedback | null
    is_retry?: bool
    step_name?: str | null

}
export interface StepInfo extends StepItem {
  name: string
  user_feedback?: string
  chunked_feedback?: string
  input?: StepInput
}

export interface UserState {
  all_steps: StepInfo[]
  next_step: StepInfo
  last_step: StepInfo
  video_id: string
  user_id: string
}

export interface UploadVideoParams {
  videoFile: File
  userEmail: string
  timelineName: string
}

export type Video = {
  filename: string
  remoteUrl: string
  hash: string
}
