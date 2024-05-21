"use client";
import {
  Step,
  Stepper,
  useStepper,
  type StepItem,
} from "@/components/ui/stepper"
import { Button } from "@/components/ui/button"
import useSWR from 'swr'
import axios from 'axios';
import React, { createContext, useContext, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL

const fetcherWithParams = async (url, params) => {
  try {
    const res = await axios.get(url, { baseURL: API_URL, params })
    console.log('fetcherwithparams data', res.data);
    return res.data;
  } catch (error) {
    throw error
  }
}

function getLatestState(userEmail: string, timelineName: string, lengthSeconds: number, videoHash: string): UserState {
  const params = {
    user_email: userEmail,
    timeline_name: timelineName,
    length_seconds: lengthSeconds,
    video_hash: videoHash,
    with_output: true,
    wait_until_done_running: false,
    block_until: false,
    timeout: 5,
  }
  const { data, error, isLoading } = useSWR(['get_latest_state', params], async ([url, params])=> {
    if (params.userEmail === '') return {}
    return await fetcherWithParams(url, params)
  })
  if (error) {
    console.error(error)
  }
  console.log('data', data);
  if (data && data.error) {
    console.error(data.error)
  } else if (data) {
    console.log('returning data');
    return data;
  }
}
interface PartialFeedback {
    partials_to_redo: Array<bool> | null
    relevant_user_feedback_list: Array<str | null> | null
}

interface StepInput {
    user_prompt?: string | null
    llm_modified_partial_feedback?: PartialFeedback | null
    is_retry?: bool
    step_name?: str | null

}
interface StepInfo {
  name: string
  user_feedback?: string
  chunked_feedback?: string
  input?: StepInput
}

interface UserState {
  all_steps: StepInfo[]
  next_step: StepInfo
  last_step: StepInfo
  video_id: string
  user_id: string
}

export default function MainStepper({ userData }) {
  //const [latestState, setLatestState] = useState<UserState>({});
  //setLatestState(initialState);
  const latestState = getLatestState(userData.email, 'timelineName', 60, '3985222955')
  console.log('latestState', latestState);
  let steps = [
    { label: "Upload video" },
    { label: "..." },
    { label: "Profit!" },
  ] satisfies StepItem[]

  if (latestState && latestState.all_steps) {
    steps = latestState.all_steps.map((step) => {
      return { label: step.name, userFeedback: step.user_feedback, chunkedFeedback: step.chunked_feedback }
    })
  }
  return (
    <div className="flex w-full flex-col gap-4">
      <Stepper initialStep={0} steps={steps}>
        {steps.map(({ label }, index) => {
          return (
            <Step key={label} label={label}>
              <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
                <h1 className="text-xl">Step {index + 1}</h1>
              </div>
            </Step>
          )
        })}
        <Footer />
      </Stepper>
    </div>
  )
}

const Footer = () => {
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    hasCompletedAllSteps,
    isLastStep,
    isOptionalStep,
  } = useStepper()
  return (
    <>
      {hasCompletedAllSteps && (
        <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
          <h1 className="text-xl">Woohoo! All steps completed! 🎉</h1>
        </div>
      )}
      <div className="w-full flex justify-end gap-2">
        {hasCompletedAllSteps ? (
          <Button size="sm" onClick={resetSteps}>
            Reset
          </Button>
        ) : (
          <>
            <Button
              disabled={isDisabledStep}
              onClick={prevStep}
              size="sm"
              variant="secondary"
            >
              Prev
            </Button>
            <Button size="sm" onClick={nextStep}>
              {isLastStep ? "Finish" : isOptionalStep ? "Skip" : "Next"}
            </Button>
          </>
        )}
      </div>
    </>
  )
}
