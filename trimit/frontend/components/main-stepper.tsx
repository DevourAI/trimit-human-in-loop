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
import { StepperForm, FormSchema } from "@/components/stepper-form"
import { createChunkDecoder, createJsonChunkDecoder } from "@/lib/streams";




const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL

const fetcherWithParams = async (url, params) => {
  try {
    const res = await axios.get(url, { baseURL: API_URL, params })
    console.log('fetcherwithparams data', res.data);
    return res.data;
  } catch (error) {
    console.error('fetcherwithparams error', error);
    return { error }
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
interface StepInfo extends StepItem {
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
  const videoHash = '3985222955'
  const timelineName = 'timelineName'
  const lengthSeconds = 60
  const latestState = getLatestState(userData.email, timelineName, lengthSeconds, videoHash)
  console.log('latestState', latestState);
  let steps = [
    { name: "Upload video" },
    { name: "..." },
    { name: "Profit!" },
  ] satisfies StepInfo[]

  if (latestState && latestState.all_steps) {
    steps = latestState.all_steps.map((step) => {
      return { label: step.name, ...step }
    })
  }

  const [activePrompt, setActivePrompt] = useState<string>('')
  const [finalResult, setFinalResult] = useState({})

  async function onSubmit(stepIndex: number, data: z.infer<typeof FormSchema>) {
    const params = {
      user_email: userData.email,
      timeline_name: timelineName,
      length_seconds: lengthSeconds,
      video_hash: videoHash,
      user_input: data.feedback,
      streaming: true,
      force_restart: true,
      ignore_running_workflows: true,
    }
    const url = new URL(`${API_URL}/step`)
    const decode = createJsonChunkDecoder();
    Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
    try {
      const res = await fetch(
        url.toString(),
        {
          method: "GET",
        }
      ).catch((err) => {
        throw err;
      });
      if (!res.ok) {
          throw new Error((await res.text()) || "Failed to fetch the chat response.");
      }

      if (!res.body) {
          throw new Error("The response body is empty.");
      }
      const reader = res.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        const valuesDecoded = decode(value);
        const newMessagePart = valuesDecoded.map((valueDecoded) => {
          if (valueDecoded && valueDecoded.message && !valueDecoded.is_last) {
            return valueDecoded.message
          }
        }).join('')
        setActivePrompt(activePrompt + newMessagePart)
        if (valuesDecoded.length > 0) {
          const lastValue = valuesDecoded[valuesDecoded.length - 1]
          if (lastValue.is_last) {
            setFinalResult(lastValue)
          }
        }
      }
    } catch (err: unknown) {
      console.error(err);
    }



    // const eventSource = new EventSource(url);
    // console.log("eventSource", eventSource);
    // eventSource.onmessage = function(event) {
        // console.log('New data:', event.data);
        // setActivePrompt(activePrompt + event.data)
    // };
    // eventSource.onerror = function(error) {
        // console.log('EventSource failed:', error);
        // eventSource.close();
    // };
  }
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    hasCompletedAllSteps,
    isLastStep,
    isOptionalStep,
    activeStep
  } = useStepper()
  console.log("activeStep", activeStep);
  console.log("activePrompt", activePrompt);

  return (
    <div className="flex w-full flex-col gap-4">
      <Stepper initialStep={0} steps={steps}>
        {steps.map(({ name, input }, index) => {
          return (
            <Step key={name} label={name}>
              <div className="grid w-full gap-2">
                <p>{index == activeStep ? activePrompt: ''}</p>
                <StepperForm stepIndex={index} onSubmit={onSubmit} userData={userData} step={steps[index]} />
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
