"use client";
import {
  Step,
  Stepper,
  useStepper,
} from "@/components/ui/stepper"
import { Button } from "@/components/ui/button"
import React, { createContext, useContext, useState, useReducer, useEffect } from 'react';
import { StepperForm, FormSchema } from "@/components/stepper-form"
import UploadVideo from "@/components/ui/upload-video"
import VideoSelector from "@/components/ui/video-selector"
import { decodeStreamAsJSON } from "@/lib/streams";
import {
  step,
  getLatestState,
  getStepOutput,
  resetWorkflow,
  revertStepInBackend,
  uploadVideo,
  downloadVideo,
  downloadTimeline
} from "@/lib/api";
import {
  type CommonAPIParams,
  type StepOutputParams,
  type GetLatestStateParams,
  type RevertStepParams,
  type ResetWorkflowParams,
  type StepInfo,
  type UserState,
  type StepParams,
  type DownloadVideoParams
} from "@/lib/types";
import {stepData} from "@/lib/data";

const BASE_PROMPT = 'What do you want to create?'

function remove_retry_suffix(stepName: string): string {
  return stepName.split('_retry_', 1)[0]
}

function stepsFromState(state: UserState): StepInfo[] {
  const defaultSteps = [
    { name: "Upload video" },
    { name: "..." },
    { name: "Profit!" },
  ] satisfies StepInfo[]

  if (!state || !state.all_steps) {
    return [defaultSteps, defaultSteps]
  }

  const allSteps = state.all_steps.map((step) => {
    return { label: step.name, ...step }
  })
  if (allSteps.length === 0) {
    return [defaultSteps, defaultSteps]
  }
  const actionSteps = allSteps.filter((step) => step.user_feedback)
  return [allSteps, actionSteps]
}

function findNextActionStepIndex(allStepIndex, allSteps, actionSteps) {
  const actionStepNames = actionSteps.map((step) => step.name)
  for (let i = allStepIndex; i < allSteps.length; i++) {
    const actionStepIndex = actionStepNames.indexOf(allSteps[i].name)
    if (actionStepIndex > -1) {
      return actionStepIndex
    }
  }
  return actionSteps.length
}

function stepIndexFromState(state: UserState): number {
  const [allSteps, actionSteps] = stepsFromState(state)
  if (state && state.next_step) {
    return stepIndexFromName(state.next_step.name, allSteps, actionSteps)
  }
  return 0
}

function stepIndexFromName(stepName: string, allSteps: StepInfo[], actionSteps: StepInfo[]): number {
  const _currentAllStepIndex = allSteps.findIndex((step) => step.name === remove_retry_suffix(stepName))
  if (_currentAllStepIndex !== -1) {
    const nextActionStepIndex = findNextActionStepIndex(_currentAllStepIndex, allSteps, actionSteps)
    if (nextActionStepIndex >= actionSteps.length) {
      console.log('All action steps completed');
    }
    return nextActionStepIndex
  } else {
    console.error(`Could not find step ${remove_retry_suffix(stepName)} in steps array ${allSteps}`)
  }
  return 0
}


export default function MainStepper({ userData }) {
  const [videoHash, setVideoHash] = useState(null)
  const [videoProcessingCallId, setVideoProcessingCallId] = useState(null)
  const [latestState, setLatestState] = useState(null)
  const [userFeedbackRequest, setUserFeedbackRequest] = useState(null)
  const [trueStepIndex, setTrueStepIndex] = useState(0)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [finalResult, setFinalResult] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [needsRevert, setNeedsRevert] = useState(false)
  const [allSteps, actionSteps] = stepsFromState(latestState)

  const timelineName = 'timelineName'
  const lengthSeconds = 60
  const userParams: UserParams = {
    user_email: userData.email,
    timeline_name: timelineName,
    length_seconds: lengthSeconds,
    video_hash: videoHash,
  }


  useEffect(() => {
    async function fetchLatestState() {
        const data = await getLatestState(userParams as GetLatestStateParams);
        setLatestState(data);
        setCurrentStepIndex(stepIndexFromState(data))
    }

    fetchLatestState();
  }, [userData, videoHash]);

  useEffect(() => {
    setUserFeedbackRequest(latestState?.output?.user_feedback_request)
    setTrueStepIndex(stepIndexFromState(latestState))

  }, [latestState]);

  useEffect(() => {
    setUserFeedbackRequest(finalResult?.user_feedback_request || BASE_PROMPT)
  }, [finalResult]);

  const [activePrompt, activePromptDispatch] = useReducer((state, action) => {
    switch (action.type) {
      case 'append':
        return state + action.value;
      case 'restart':
        return action.value;
      default:
        throw new Error('Unhandled action type');
    }
  }, '');

  async function onSubmit(stepIndex: number, data: z.infer<typeof FormSchema>) {
    console.log("submitting")
    console.log("feedback", data.feedback)
    setIsLoading(true)
    if (needsRevert) {
      console.log('reverting step');
      for (let i = 0; i < trueStepIndex - stepIndex; i++) {
        await undoLastStepBeforeRetries()
      }
      setNeedsRevert(false)
    }
    const params = {
      user_input: data.feedback || '',
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      ...userParams,
    } as StepParams
    await step(params, async (reader) => {
      activePromptDispatch({ type: 'restart', value: '' });
      const lastValue = await decodeStreamAsJSON(reader, (value) => {
        let valueToAppend = value;
        if (typeof value !== 'string') {
          if (value.name !== undefined) {
            const stepIndex = stepIndexFromName(value.name, allSteps, actionSteps)
            setTrueStepIndex(stepIndex)
            setCurrentStepIndex(stepIndex)
            valueToAppend = ''
          } else if (value.chunk !== undefined && value.chunk === 0) {
            if (typeof value.output === 'string') {
              valueToAppend = value.output;
            } else {
              console.log('value.output is not a string', value.output)
              valueToAppend = ''
            }
          } else if (value.message !== undefined) {
            if (typeof value.message === 'string') {
              valueToAppend = value.message;
            } else if (value.message.output && typeof value.message.output === 'string') {
              if (value.message.chunk === 0) {
                valueToAppend = value.message.output
              } else {
                valueToAppend = ''
              }
            } else {
              console.log('value.message.output is not a string', value.message.output)
              valueToAppend = ''
            }
          } else {
            valueToAppend = ''
          }
        }
        activePromptDispatch({ type: 'append', value: valueToAppend });
      });
      if (!typeof lastValue === 'string') {
        console.log('lastValue is not a string', lastValue)
      } else {
        setFinalResult(lastValue)
      }
      setIsLoading(false)
    })
  }



  async function restart() {
    setIsLoading(true)
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({})
    await resetWorkflow(userParams as ResetWorkflowParams)
    const newState = await getLatestState(userParams as GetLatestStateParams)
    setLatestState(newState)
    setCurrentStepIndex(0)
    setIsLoading(false)
  }

  async function revertStep(toBeforeRetries) {
    // revertStepInBackend should take a step name or index as a parameter and do multiple reverts if needed
    setIsLoading(true)
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({})
    await revertStepInBackend({to_before_retries: toBeforeRetries, ...userParams} as RevertStepParams)
    const latestState = await getLatestState(userParams as GetLatestStateParams)
    setLatestState(latestState)
    setCurrentStepIndex(stepIndexFromState(latestState))
    setIsLoading(false)
  }
  async function undoLastStep() {
    await revertStep(false)
  }

  async function undoLastStepBeforeRetries() {
    await revertStep(true)
  }

  async function prevStepWrapper() {
    if (currentStepIndex === 0) {
      return
    }
    setCurrentStepIndex(currentStepIndex - 1)
    const currentStepName = actionSteps[currentStepIndex].name
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult(await getStepOutput(
      { step_keys: currentStepName, ...userParams}
    ))
    setNeedsRevert(true)
  }

  async function uploadVideoWrapper(videoFile) {
    const respData = await uploadVideo({videoFile, userEmail: userData.email, timelineName})
    console.log("upload response data", respData)
    if (respData && respData.videoHash) {
      console.log("got video hash", respData.videoHash)
    }
    if (respData && respData.callId) {
      setVideoProcessingCallId(respData.callId)
    }
  }

  return (
    <div className="flex w-full flex-col gap-4">
       <UploadVideo uploadVideo={uploadVideoWrapper}/>
       <VideoSelector userData={userData} setVideoHash={setVideoHash}/>
       <Stepper initialStep={currentStepIndex} steps={stepData.stepArray}>
         {stepData.stepArray.map(({ name, human_readable_name }, index) => {
           return (
             <Step key={name} label={human_readable_name}>
               <div className="grid w-full gap-2">
                 <StepperForm
                   systemPrompt={activePrompt}
                   undoLastStep={undoLastStep}
                   isLoading={isLoading}
                   prompt={userFeedbackRequest}
                   stepIndex={index}
                   onSubmit={onSubmit}
                   userData={userData}
                   step={stepData.stepArray[index]}
                   remoteStepData={actionSteps[index]}
                 />
               </div>
             </Step>
           )
         })}
  {/*<Footer currentStepIndex={currentStepIndex} prevStepWrapper={prevStepWrapper} restart={restart} />*/}
       </Stepper>
       <Button onClick={() => downloadVideo({user_email: userData.email, timeline_name: timelineName, video_hash: videoHash, length_seconds: lengthSeconds} as DownloadVideoParams)}>
          Download latest video
       </Button>
       <Button onClick={() => downloadTimeline({user_email: userData.email, timeline_name: timelineName, video_hash: videoHash, length_seconds: lengthSeconds} as DownloadTimelineParams)}>
          Download latest timeline
       </Button>
    </div>
  )
}

const Footer = ({restart, prevStepWrapper, currentStepIndex}) => {
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    hasCompletedAllSteps,
    isLastStep,
    isOptionalStep,
    setStep
  } = useStepper()
  setStep(currentStepIndex)
  return (
    <>

      {hasCompletedAllSteps && (
        <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
          <h1 className="text-xl">Woohoo! All steps completed! 🎉</h1>
        </div>
      )}
      <div className="w-full flex justify-end gap-2">
          <>
          <Button size="sm" onClick={restart}>
            Restart
          </Button>
          <Button
            disabled={currentStepIndex === 0}
            onClick={prevStepWrapper}
            size="sm"
            variant="secondary"
          >
            Prev
          </Button>
          <Button size="sm" onClick={nextStep}>
            {isLastStep ? "Finish" : isOptionalStep ? "Skip" : "Next"}
          </Button>
          </>
      </div>
    </>
  )
}


